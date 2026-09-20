# Contract: Model Pages, Scenario Policy and Prediction Exports

**Status**: v1 plus approved visual hierarchy, color and readability/conclusion amendments implemented.  
**Modules**: `scenario_policy.py`, `salary_inference.py`, `src/pages/model_evidence.py`; page signatures remain `render(st, root, role="admin")`.

## 1. Shared policy

`scenario_policy.json` contains `schema_version=1`, policy hash, DEV snapshot identity, metadata enum identity, `pair_source=observed_dev`, allowed category/title pairs with support counts, and title-level `experience_min`, `experience_max`, `experience_median`, support count. Bounds use DEV input fields only; no actual salary, model predictions or test statistics. All three values must be finite and min ≤ median ≤ max. Preserve decimal medians; default experience widget step is 0.5 for this data, with defaults/endpoints preserved exactly. If min=max, render a fixed labelled value rather than an invalid zero-width slider.

Canonical category/title spellings must be in compatible metadata enums. No string guessing, fuzzy matching, inferred taxonomy or whole-data pair expansion. A title can be observed under multiple categories; show `Observed DEV association` and support count, not logical validity.

### Validation matrix

| Input/source | Default rule | Allowed exception |
| --- | --- | --- |
| Manual scenario | Known category/title, observed pair, finite non-boolean numeric years within title bounds | None |
| Manifest benchmark | Exact immutable record identity/profile; known enums/pair; finite years within title bounds; actual salary from artifact only | Acknowledged experience-only deviation within 0–15; otherwise reject |
| Default growth point | Observed pair; finite years within title bounds and requested graph domain | None |
| Extended growth point | All categorical/model checks plus finite years within 0–15 | Explicit acknowledged experience-only deviation; no other bypass |
| Unknown pair/enum/title | Reject before any prediction | Never |
| Missing/invalid policy/model or malformed actual | Reject dependent operation | Never |
| Queue size 0 or >100 | Run disabled / reject | Never |

Exception controls default off. Acknowledgement is bound to source type, immutable benchmark or curve request, active evidence ID and policy hash. Only experience support may be excepted; the input is not 'valid in-distribution'. Use fixed reason `experience_outside_observed_dev_range`; no free-text reason. Label `Extrapolation — outside observed DEV experience bounds` with the actual limits and `Historical q90 does not establish coverage here`. Turning exceptions off invalidates results and removes/rejects incompatible queue rows visibly; never silently modify years.

All entry paths and the final pre-inference boundary use this same validator. Fail an entire invalid batch before `predict`, identify row IDs/reasons and retain queue for correction; do not partially update results.

## 2. Logical interfaces (not new web endpoints)

Keep plain functions/simple dataclasses rather than a framework or generic repository abstraction:

- `build_scenario_policy(dev, metadata) -> policy`: no fit or target read; records DEV identity.
- `validate_scenarios(rows, policy, benchmarks, context) -> validated_rows`: raises a typed validation error with row IDs, code and message; no inference or mutation.
- `load_evidence(workspace) -> evidence`: verifies the pointer/manifest/schemas/hashes; returns explicit component availability. Only trusted-local allowlisted bundle references may deserialize.
- `predict_scenarios(validated_rows, bundle, metadata) -> result_rows`: verifies exact ordered two-feature serving contract, predicts once for the batch, checks finite shape/output, attaches bounds and audit-only actuals. No fitting/tuning or artifact writes.
- `build_growth_inputs(snapshot, policy, extend=False, acknowledgement=None) -> validated_points`: strictly bounds default grid and labels allowed extensions.
- `build_prediction_audit_csv(results) -> bytes` and `build_source_schema_csv(rows) -> bytes`: pure versioned export functions with no on-disk writes.
- Pure chart/view builders return Plotly figures/DataFrames from validated tables. UI code renders them; the offline producer may save the same figures as HTML without launching Streamlit.

Error codes: `EVIDENCE_MISSING`, `EVIDENCE_INCOMPATIBLE`, `INVALID_SCENARIO`, `EXPERIENCE_OUT_OF_BOUNDS`, `ACKNOWLEDGEMENT_REQUIRED`, `QUEUE_LIMIT`, `MODEL_SCHEMA_MISMATCH`, `PREDICTION_INVALID`. Messages identify the corrective action; never hide an invalid path/model behind a generic successful fallback.

## 3. Session state and reactive controls

- Scope all state and caches by resolved workspace, active evidence ID and policy hash. Never trust an arbitrary browser-supplied root/path; consume the router's selected root and validate resolution.
- Page state: queue rows, queue revision, next scenario ordinal, result snapshot, exception acknowledgements and builder selections. No module-global per-user mutable state.
- Category/title/experience controls live outside `st.form` to update immediately. On category change, preserve title only if still allowed; otherwise select the first allowed title. On title change, reset experience to that title's median. Updates occur in callbacks/before widget creation.
- Add/quick-load appends a validated unique scenario and increments revision; duplicates retain distinct IDs. Every mutation clears the prior result snapshot and curve results. Editing builder controls alone does not modify queued records.
- Successful Run binds results to `(workspace, evidence_id, policy_hash, revision, model_id)`; display/download only while this tuple still matches.
- Clear removes queue, results, curves and acknowledgements. Workspace/evidence/policy change performs the same invalidation and explains why. Old queue formats must not silently migrate into the new serving contract; reset with a message.
- Run is disabled for an empty/invalid queue or incompatible/missing Top-2 artifact. Page 06 must not load the full bundle as a two-input fallback.
- Growth grid defaults to integer years within each title's supported range intersected with 0–15 plus exact in-domain endpoints; it uses the successful queue's distinct `(category,title)` profiles. Curves overlap where category matches because title is descriptive only. Optional extended points remain separate, dashed/warning-labelled and explicitly requested, not recomputed on every unrelated rerun.

## 4. Calculation and labels

- All arithmetic uses unrounded finite values. MAE/RMSE/MedAE are USD; R² and error ratio are unitless; times are seconds.
- MAE gap: `validation - train`; explosion ratio: `validation/train` only when train >0.
- CV-to-test absolute delta: `test - cv`; percentage delta: `100*(test-cv)/abs(cv)` when cv ≠0. Label metric and direction. R² uses its own unitless panel; do not share a dollar scale. Never call lower MAE a percentage accuracy gain.
- Diagnostic badge rules: missing required comparable metrics → `Insufficient evidence`; Dummy → `Reference baseline`; candidate validation MAE below matching Dummy MAE → `Below baseline`; otherwise → `At/above baseline`. Gap/ratio displayed separately, not a three-way fit classifier. Missing Dummy means baseline badge unavailable, not assumed $54,011.
- Point/bounds: `lower=max(0,pred-q90)`, `upper=pred+q90`, with nonnegative finite q90 from matching model evidence (zero is a valid degenerate empirical band). If a finite prediction produces impossible interval ordering (e.g. upper<lower), reject the result rather than force a misleading chart.
- Residual: actual−predicted. Absolute error: abs(predicted−actual). Signed variance: `100*(predicted-actual)/actual` only for positive known actual. Zero actual gives an absolute error but blank percentage with a reason; unknown actual gives no actual marker or errors. Negative/nonfinite supplied actuals are invalid.
- Charts display whole USD, R² to three decimals, percentage to one decimal; tables can show the same precision with full evidence downloadable. Actual markers retain their true location even outside the empirical band. Explain zero-clipped lower intervals.
- Family selection, tuned full-model selection and fixed Top-2 serving are separate labels/IDs. A new candidate rank cannot overwrite the saved selected-family chip on Page 05.

## 5. Page contract and missing states

### Page 04

Five KPI chips, the metric guide and a high-priority collapsed temporal-validation guide precede four tabs: `Overall comparison`, `Random Forest`, `Gradient Boosting`, `Baseline controls`. Always identify evidence/configuration scope. The guide explains the actual adjacent-block temporal splitter, fold-local preprocessing, purpose, metrics and limitations from current fold evidence; it is not generic randomized or expanding-window K-fold copy. Overall combo → compact emphasized decision table → ratio/R² combo → detailed executive evidence → recorded fold timeline. RF fold chart/drift precede evidence table/chips. GB head-to-head precedes MedAE table. Baselines contrast actual available models; missing family/tab data receives a local message, not a substituted model. No invariant that RF must rank first.

### Page 05

Six KPI chips (model, test R², MAE, MedAE, named R² gap, q90), actual count/period, then five sections: generalization, diagnostic scatter/residuals, tuning, feature reliance/variant comparison, uncertainty/audit. The tuning section's high-priority methodology guide defaults collapsed and appears before sensitivity charts. It describes the four-anchor/coordinate-sweep implementation, R² ranking, carried/fixed values, saved-configuration source and non-nested limitation. Tuning charts precede summary detail; applied settings and sweep winners differ when the evidence says so. Dense scatter uses hover rather than overlapping labels. Historical examples carry inclusion/exposure labels and no guaranteed-accuracy verdict.

### Page 06

Metadata-driven Top-2 strip then benchmark, builder, queue/actions, results. Use three buttons only when three verified examples exist; otherwise show available count/reason. No hardcoded sample title/actual. Results show salary/whisker chart first, then requested growth view, then table/downloads. Prediction chart page size is 10; show `Scenarios a–b of n`. The full result table and CSV retain all n (≤100) rows.

Unavailable model evidence must not crash the page or invent defaults. New pages consume the explicit supplemental manifest only; they do not silently blend similarly named legacy CSVs. Remediation cites the offline CLI. Missing optional tuning data must not disable Page 06 or unrelated valid charts.

All controls have accessible labels; bar/line/whisker values are legible without hover at agreed desktop widths. Colors also differ by legend/shape/text. No new CSS, custom component, login mechanism or navigation system is required.

### Shared table grammar and audit downloads

- Compact decision tables use `st.table` with Markdown: `**bold**` means the governing winner, selected/applied value or consequential comparison; `*italic*` means reference/baseline/context. Every such table shows the legend `**Bold = governing result** · *Italic = reference/context*`.
- Wide or complete evidence uses `st.dataframe`; cells contain plain values, never raw Markdown. `column_config` owns numeric formatting and reader-facing labels. Identity columns are pinned where supported.
- Display formats are whole USD, R² to three decimals, percentages to one decimal, seconds to three decimals and counts as integers. Highlighting uses unrounded values. Equal best values receive equal emphasis/rank; display rounding cannot select a unique winner.
- Page 04 family decisions use minimum mean validation MAE. Dummy is a reference even if its numeric rank changes. Page 05 inherited tuning stages use maximum CV R²; saved applied settings come from the manifest/bundle and are not inferred from sweep styling.
- Current supplemental files are authoritative. Optional training-audit downloads are restricted to `outputs/08_full_pipeline/logs/` and require a complete manifest, matching source SHA-256, contained nonsymlinked paths and verified checksums. Show audit/pipeline run IDs and label the trace `historical primary-pipeline audit`; never merge it into active-pack calculations.
- Page 04 exports active candidate/fold summary and membership. Page 05 exports the tuning summary and each available sweep. A compatible audit trace additionally offers its manifest, relevant fold/tuning CSVs and complete JSONL. Missing logs are an optional unavailable state and never trigger training.

### Shared semantic chart palette

Colors are assigned by evidence role, never by trace order or whichever model ranks first:

| Evidence role | Required color family | Required non-color encoding |
| --- | --- | --- |
| Validation error / primary prediction | Blue or teal (`#1976D2` / `#00897B`) | Column/bar plus direct value label |
| Training error / DEV-CV comparator | Orange (`#F28E2B`) | Line plus circular markers and labels |
| Historical-test error | Red (`#D32F2F`) | Column/bar plus `Historical test` legend text |
| Dummy/failure/threshold reference | Dark red (`#B71C1C`) | Dashed horizontal line plus annotation |
| Known actual salary | Red accent (`#C62828`) | Diamond marker plus `Actual` legend text |
| Empirical uncertainty | Amber (`#F9A825`) | Whisker/error-bar geometry plus endpoint labels |
| Neutral/reference/disabled evidence | Gray (`#6B7280`) | Explicit reference/unavailable text |

Red historical-test columns communicate that test error is the consequential out-of-time error, not that the result automatically failed. Native `st.error`/`st.warning` remains reserved for blocking/error and caution messages. Green is reserved for verified success/status badges, not a scientific “good fit” conclusion.

### Chart priority and width contract

- **P1 — decision chart**: the first chart that directly answers the section question. Render first; `width="stretch"` is allowed within the full main-content container. At most one P1 chart per analytical section.
- **P2 — supporting comparison**: corroborates or decomposes P1. Place in a bounded container (target 680–900 px) or one side of a two-column desktop row. It may stretch only inside that container. On narrow viewports, native stacking is allowed.
- **P3 — diagnostic/reference**: detailed residual, ECDF, encoded-importance or secondary sensitivity evidence. Place in a compact column, a two-column grid, a collapsed expander or an explicit on-demand view. It must not automatically consume full content width.
- **Promotion rule**: promote P2/P3 one tier only when direct labels, axis categories or uncertainty endpoints would otherwise clip/overlap at 1280 px. Record the reason in a figure/layout test. Do not truncate required table/export evidence to keep a chart compact.
- **Page mapping**: Page 04 overall train/validation comparison is P1; ratio/R² is P2; timeline/drift are P3. Page 05 dollar-error generalization is P1; separate R², residual, ECDF and encoded importance are P2/P3; tuning sensitivities use a two-column P2 grid. Page 06 prediction-with-whiskers is P1; growth curves are P2 and only render on request.

### Readability profile contract

Each figure builder returns a figure whose layout already contains its readability geometry. `show_plot` may add responsive config/hover defaults but must not overwrite explicit margins, height, tick formats or automargins.

| View | Minimum profile | Direct-label policy | Dense fallback |
| --- | --- | --- | --- |
| Page 04 candidate comparison | P1, height ≥460, bottom automargin for five model names | all validation bars; train line only where collision-free | full train values in table/hover |
| Page 04 ratio/R² | P2, height ≥420 | ratio bars; R² key points | table holds all exact values |
| Page 04 RF folds / GB head-to-head | P1 within tab, height ≥440 | validation bars; selected train points | annotate worst fold and use hover for other line values |
| Page 04 drift | P3, height ≥440 | no per-point feature text by default | hover plus labelled Top-3 callouts/table |
| Page 05 dollar errors | P1, height ≥430 | all three historical-test bars and DEV line where separated | conclusion/table repeats exact deltas |
| Page 05 tuning | P2, each width ≥480 and height ≥380; stack instead of two columns below threshold | bar labels plus selected optimum R² only | full sweep table/hover |
| Page 05 horizontal importance | P1/P3, height based on visible row count; left automargin | bar values for visible Top N | full evidence download |
| Page 05 scatter/residual/ECDF | diagnostic profile | no dense point text; summary annotation only | hover and conclusion |
| Page 06 prediction | P1, height ≥460; max 10 short scenario IDs | prediction labels and actual labels; endpoints only if collision-free | bounds in hover/table |
| Page 06 growth | P2, height ≥420 | endpoint labels only when distinct | full series in hover |

Axes use comma-grouped whole USD or disclosed `$k` ticks, never ordinary scientific notation. Categorical axes use full values when they fit; deterministic display labels must map back to full source labels. Horizontal feature charts reserve left margin according to longest visible label. `cliponaxis=False` may be used for required outside labels only when the surrounding margin accommodates them.

### Report brief and conclusion contract

A page brief contains:

```text
question, evidence_scope, takeaway, primary_limitation
```

A conclusion block contains:

```text
conclusion_id, evidence_id, model_id_or_scope, population,
finding, why_it_matters, limit, decision_or_use, source_fields
```

Conclusion builders are pure and receive validated DataFrames/records. They do not read page state, invent thresholds or use model names/numbers from prose. Required source fields missing/nonfinite produce `Conclusion unavailable — missing <fields>` and the existing evidence remediation. Visible conclusion order is fixed:

1. **Finding** — direct observed result with values and units.
2. **Why it matters** — bounded practical interpretation.
3. **Limit** — most decision-relevant scientific/provenance caveat.
4. **Decision/use** — optional safe next use, never an automated model promotion or compensation recommendation.

Metric glossary copy is stable educational text, while takeaways/conclusions are evidence-derived. Required warnings remain outside collapsed containers.

## 6. Prediction audit CSV v1

Filename: `salary_prediction_audit_v1.csv`; UTF-8; comma delimiter; header; no index. Exact column order:

```text
schema_version,scenario_id,source,record_id,evidence_id,model_id,feature_variant,policy_hash,job_title,job_category,years_of_experience,validation_mode,exception_reason,predicted_salary_usd,lower_bound_usd,upper_bound_usd,q90_abs_error_usd,q90_basis,actual_salary_usd,absolute_error_usd,signed_variance_pct
```

- `schema_version=1`; `source=manual|benchmark`; `feature_variant=top2`; `validation_mode=strict|experience_exception`.
- Model/evidence/policy IDs must match the displayed snapshot. Manual `record_id` blank; source benchmark ID is not a new synthetic job ID.
- USD values export to two decimals, signed variance to four decimals, years retain supplied precision; rounding applied only after arithmetic. Missing values are empty cells, not zero or the string `nan`.
- No field named `error_pct` is silently redefined. UI copy explains that this new audit export uses **signed** variance, whereas the older result export used absolute percentage error.
- Exceptions retain fixed reason/validation mode. The associated on-screen/download caption states empirical bands are historical reference only for extrapolation; the flags make that warning traceable after export.
- Growth curves are a labelled diagnostic view, not an additional download in this feature. Never append derived curve points to the scenario queue or its exports.

## 7. Original enterprise-schema scenario CSV

Filename: `salary_scenarios_source_schema_v1.csv`. Exactly `SOURCE_COLUMNS`, in this order:

```text
job_id,job_title,AI Engineering,experience_level,years_of_experience,education_required,annual_salary_usd,salary_min_usd,salary_max_usd,city,country,remote_work,company_size,industry,required_skills,ai_salary_premium_pct,demand_score,demand_growth_yoy_pct,benefits_score_10,posting_year,posting_month,is_senior,is_remote_friendly,is_llm_role,salary_tier
```

Map category to `AI Engineering`. Manual rows populate `job_id=SCENARIO-<scenario_id>`, title/category/years; unknown original attributes remain blank, especially salary and posting dates. Benchmark rows may preserve verified source fields; where original job_id was dropped, use the clearly synthetic scenario ID and retain true evidence-row identity only in the audit CSV. Never search the raw file for a guessed matching original ID or manufacture country/skill values.

This file preserves schema shape for interchange; incomplete scenarios are not a valid training dataset. Predictions/bounds/errors are in the audit CSV, never substituted for source actual/min/max salary. Different export contracts must not overwrite or masquerade as each other.

## 8. Acceptance test boundary

Pure pytest covers all formulas, matrix branches, identities and export bytes. Estimator spies assert invalid input produces zero predict calls, valid batch exactly one batch call (growth requested separately), and all serving actions zero fit/tune calls. AppTest exercises real `streamlit.py` routing with controlled login/workspace fixtures, page-specific reruns, disabled controls, state invalidation and missing packs. Browser review verifies actual labels/whiskers/scrolling/keyboard/download behavior; it cannot be marked passed based only on AppTest.
