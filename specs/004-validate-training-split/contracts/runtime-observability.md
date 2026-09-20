# Contract: Runtime, Metrics and Human/Agent Evidence

**Status**: Implemented and fixture-verified; no current real-data benchmark or training pack was generated.  
**Parent**: [training-evidence.md](training-evidence.md), unpublished `training-validation/v1`.  
**Scope**: Offline measurement and read-only export contracts, not UI implementation, live log streaming, production monitoring or automatic model optimization.

## 1. Questions the evidence must answer

1. Which model has the best validated accuracy, which is fastest, and why was the selected family chosen?
2. Where did run time go: fitting, tuning, prediction, diagnostics, measurement overhead or publication?
3. What completed, failed, was skipped or reused; what action is safe next?
4. Can a person, an agent and a future UI obtain the same answer from the same immutable evidence without parsing English prose or retraining?

“Best performance” is multidimensional. Keep distinct fields for `lowest_cv_mae_model_id`, `selected_family_id` (existing MAE/overlap/simplicity rule), `fastest_fit_model_id`, `fastest_warm_predict_model_id` and `pareto_model_ids`. No speed-weighted score, changed winner or production-readiness claim is introduced. A model on the Pareto frontier need not be the selected model.

## 2. Accuracy and stability metric dictionary

`metric_catalog.json` contains metric_id, English name/definition, unit, direction (`lower`, `higher`, `context_only`), aggregation, valid scopes, null reasons, display precision and caveat. A value is never interpreted from a column label alone.

| Metric ID | Definition and permitted scope | Interpretation |
| --- | --- | --- |
| `mae_usd`, `rmse_usd`, `medae_usd`, `r2` | Existing definitions; paired TRAIN/outer validation and final holdout | MAE governs family ranking; null R² for n<2 or constant target |
| `mean_residual_usd` | mean(actual - predicted) | Positive means underprediction; signed, not a quality rank |
| `p90_abs_error_usd`, `p95_abs_error_usd` | Linear-interpolated quantiles of absolute residuals from already generated predictions | Tail-error diagnostics; not confidence bounds on a mean |
| `mae_skill_vs_dummy_pct` | 100 × (Dummy MAE - model MAE) / Dummy MAE, matched population | Higher better; null `zero_baseline` if Dummy MAE=0 |
| `train_validation_mae_gap_usd` | validation MAE - TRAIN MAE for the same fitted fold model | Positive gap alone does not prove overfitting |
| `normalized_mae_gap` | gap / matched Dummy validation MAE | Existing 0.20 heuristic; null for zero baseline |
| `r2_train_minus_validation` | TRAIN R² - validation R² | Null if either undefined; context only |
| `cv_mae_mean_usd`, `cv_mae_sd_usd`, `cv_mae_min_usd`, `cv_mae_max_usd` | Equal-weight five-fold MAE mean, population SD, minimum and maximum | Do not call five folds independent repeated trials or SD a confidence interval |
| `valid_fold_count`, `failed_fold_count` | Count complete/failed candidate-fold results | Expected five; incomplete candidates cannot win |
| `worst_fold_id` | Largest validation MAE; earliest fold breaks ties | Cite its month/support, not just a rank |
| `pooled_oof_*` | Core/tail/bias metrics from concatenated distinct outer-validation predictions | Different population weighting from equal-fold means; never replace selection metric |
| `tuning_mae_gain_usd` | factory RF outer MAE - nested tuned-full outer MAE, matched folds | Positive improvement; report five paired deltas/mean, no significance claim |
| `top2_minus_full_mae_usd` | Top-2 minus full MAE on matched folds or the final holdout | Positive means Top-2 worse; no holdout-driven promotion |

Store per-fold values and sample counts before aggregates. Keep full precision; human displays use whole USD by default, R² to three decimals, percentages to one decimal. Include raw exports. No new fitting or prediction is needed for these derived metrics. Persist row-level OOF predictions for candidate/full/Top-2 comparison to make pooled/tail metrics independently reproducible, bound to row and fit IDs; never include reserve rows.

Invalid/undefined values are null plus reason, not zero/NaN/Infinity. Predictive failures block the corresponding result; a failed fold is not excluded to improve the mean. If some diagnostic R² values are undefined, report the aggregate null with valid-count context rather than silently averaging a different subset.

## 3. Timing boundaries and cost accounting

Use `perf_counter_ns` for wall time and `process_time_ns` for process CPU deltas. Export seconds, keeping integer ns samples where needed to audit fast calls. Record operation ID, parent ID, model/configuration/feature/fold/search identity and work count with every timing. UTC timestamps are for correlation, not duration arithmetic.

| Measurement | Boundary / rule |
| --- | --- |
| `fit_wall_s`, `fit_cpu_s` | One `Pipeline.fit`: includes existing preprocessing fit/transform + estimator fit; no pretend isolated preprocessing measurement |
| `predict_wall_s`, `predict_cpu_s` | One `Pipeline.predict`: includes preprocessing transform + estimator prediction; model already loaded |
| `search_wall_s` | Entire bounded search including scoring/bookkeeping; trial fit totals are children, not additional elapsed time |
| `stage_wall_s`, `run_wall_s` | Inclusive stage/run spans; report exclusive phase categories separately where spans do not overlap |
| `scoring_wall_s`, `diagnostics_wall_s` | Metrics and importance work separated from ordinary prediction calls |
| `benchmark_overhead_s` | Extra latency measurement calls/warmups/load measurements; not scientific fitting cost |
| `serialization_wall_s`, `bundle_load_wall_s` | Actual write/read of the final trusted bundle; clearly distinguish from fit and prediction |
| `evidence_write_wall_s`, `report_wall_s` | Numeric/event writing and report/chart generation |
| `bundle_bytes`, `pack_bytes` | Actual saved sizes, not estimated in-memory model size |
| `process_peak_rss_bytes` | Linux process-lifetime high-water RSS from `resource.getrusage`; unit normalization recorded. Not per-model peak memory; does not reset per fold |
| `rss_at_operation_start_bytes`, `rss_at_operation_end_bytes` | Optional Linux process RSS samples where readable; difference is not peak allocation. Unsupported -> null with reason |

Run elapsed time is measured, not obtained by adding overlapping parent and child durations. Publish accounted nonoverlapping phase time plus `unattributed_wall_s` with method/tolerance. Run work duration ends before serializing its own terminal event; final telemetry commit overhead is explicitly excluded, avoiding circular totals. CPU time can differ from wall time and is never labelled CPU utilization percentage. Memory is contextual; do not rank models by process-lifetime high-water marks.

No extra fit is permitted solely to benchmark training speed. Five fold fits have different expanding training sizes: report mean/median/min/max and every raw fold time; do not present a p95 over five unequal workloads as repeated-fit latency. Within-fold comparisons share identical rows. Reused fits carry original measured time/source-run/hardware ID separately from current lookup time; never label reuse as a fresh zero-second fit.

## 4. Controlled prediction microbenchmark

Benchmark all 25 factory candidate fold pipelines immediately after their required fold scoring, and the two final full/Top-2 bundles after TRAIN fitting/serialization, before holdout access. No extra estimator fits or reserve/holdout reads. Final frozen factory diagnostics need not have separate microbenchmarks; missing final-factory latency is not copied from fold models.

- Input rows: first `min(100, n_fold_training)` rows by stable chronological/source order within that fitted model's TRAIN population, raw feature columns only. Final variants use the same source row IDs from full TRAIN and their respective declared column subsets. No salaries are needed. In-sample records are suitable for a compute microbenchmark, not an accuracy estimate.
- Batch sizes: 1 and `min(100, training_rows)`, deduplicated if equal; never pad or sample with replacement to pretend a 100-row workload exists.
- Three untimed-for-statistics warmup calls per model/batch, followed by 30 individually timed calls. Include warmup wall cost in overhead. Keep every measured duration; no outlier trimming. Record GC policy (normal/enabled), Python/library/thread versions and OS/CPU/logical-core count.
- `n_jobs=1`; bound numerical-library thread pools to one through the existing scikit-learn dependency `threadpoolctl` if available in the locked environment. Record effective pools; do not install new packages. If thread conditions cannot be verified, mark cross-model speed comparison non-comparable, not silently equal.
- Candidate order follows a deterministic seeded rotation across folds; record actual order. Sequential execution on the same host; load-average/context if available. No claim this removes all thermal/OS/cache variation.
- Emit min/median(p50)/p90/p95/max, mean, population SD and sample_count=30 for each exact `(model, fold, batch_size, environment, method)` group. Quantiles use linear interpolation. Do not average batch-size groups or pool fold models into one request-latency percentile.
- Throughput = total predicted rows across measured calls / sum of measured call seconds. Amortized milliseconds/row = total seconds ×1000 / total rows. Neither is p95 single-row latency. If duration is below clock resolution, mark unavailable/insufficient resolution rather than report infinite throughput.
- Final bundle load: one timed `joblib.load` of each just-produced, checksum-verified trusted local bundle into a new object. Label `first_load_in_process` and OS cache state `uncontrolled`; **not** cold disk/process/server startup latency. Time the first prediction separately, before microbenchmark warmup, as `first_predict_after_load`. No load p95 from one observation.

Bound: 27 contexts × at most two batch sizes × (3+30) = **1,782 additional predict calls**, plus two first-predict-after-load calls = **1,784**. No additional fits; the 540-fit ceiling is unchanged. Size cap 100 rows per call. Benchmark contexts/predict calls are counted separately from scoring/permutation calls. The first-predict calls use batch size 1. Exact reuse of a complete pack performs zero load/predict/benchmark calls.

A prediction microbenchmark is not live Page 06/UI/network latency or a production SLA. It is a reproducible local pipeline-cost measurement. No actual UI benchmark is in scope.

## 5. Comparison, performance budgets and tuning cost

`performance_comparison.csv` reports each family's CV accuracy/stability, fit sum/median, batch-1 warm prediction p95 by fold, actual larger batch size/throughput, selected flag, comparability and source IDs. For an optional descriptive scalar, show **median of five fold-specific batch-1 p95 values**, labelled exactly that; never label it an overall p95. `accuracy_runtime_tradeoff.csv` marks the Pareto frontier over `(cv_mae_mean_usd, total_cv_fit_wall_s)` using unrounded values: A dominates B if A is no worse in both and strictly better in at least one. Same run/input/folds/thread/hardware/method required. Missing or incompatible measurements -> frontier unknown, not dominated.

Report fastest measured candidate separately, along with accuracy loss/gain versus selected family. Do not replace MAE-based family choice with runtime or select an alternative after seeing holdout scores. Tuning cost includes slots, actual/reused/failed fits, search time and paired outer-MAE improvement; zero/impaired cost denominators are null. A negative improvement is reported honestly, not hidden as a successful tuning step. Preserve the frozen final-selection procedure.

Optional pre-run operational limits in approval:
- `max_run_wall_s`: observed work-duration budget;
- `max_final_batch1_p95_ms`: applies to each final full/Top-2 warm benchmark;
- `max_bundle_bytes`: applies to each final bundle.

Each is positive finite and includes its scope/method version. Default unset. Budget verdicts: pass / fail / not_configured / unavailable, separate from scientific Good/Bad. A violated configured operational limit means `deployment_review_eligible=false`, but does not change the scientific winner. Missing configured evidence blocks eligibility; no limits means operational readiness `not_assessed`, not “fast enough”. Combined eligibility requires scientific Good and all three operational limits configured, available and passing; still only a human-review recommendation, never production certification. These are observed budget checks, not hard watchdog/timeouts or fabricated latency targets; an in-progress call is not killed without a separately approved cancellation design.

## 6. Human-readable and agent-readable log views

Use a single structured event source; render `training.log` (plain English chronological transcript), `report.md` (narrative) and `agent_summary.json` from the same validated records. Do not independently recompute prose numbers. Store metric definitions in `metric_catalog.json`. Final outputs bind run, dataset, split, policy, method, model and environment identities.

Each event extends the parent contract with:
- `invocation_id` (new per CLI invocation), `event_id` (`invocation_id:sequence`), `operation_id`, `parent_operation_id`;
- `level`: debug/info/warning/error; `event_type` stable enum and schema version;
- `message`: short English, `reason_code`: stable machine enum, `details`: allowlisted structured fields;
- `metrics`: list of metric_id/value/unit/scope/sample_count/reason plus table row references, never embedded full tables;
- `progress`: stage-local planned/completed/failed/skipped/reused units and unit type, plus actual fit/benchmark call counts;
- `duration`: wall/CPU scope; `evidence_refs`: contained artifact path + row/filter key;
- `next_action`: safe action code, explanation and explicit approval/data prerequisite.

Progress denominators use the declared stage plan and may shrink only via a logged `stage_plan_revised` event (e.g. RF tuning skipped). They distinguish 26 trial slots from actual fits. No unreliable ETA, random training “accuracy”, unreported denominator change or time-based fake percentage. Console `concise` vs `verbose` affects rendering only; full structured events and safety warnings are retained. Cap inline details at 16 KiB and 100 metric entries; larger arrays go in checksum-linked artifacts. Flush events at operation boundaries, not per estimator tree or raw record.

Example English **template, not measured evidence**:

```text
[INFO] Candidate <model>, fold <k>/5 completed.
Train: <rows> rows (<months>); validation: <rows> rows (<month>).
MAE: <value> USD; Dummy MAE: <value> USD; improvement: <value>%.
Pipeline fit: <seconds>s; validation prediction: <seconds>s.
Finding: <derived statement>. Limit: <fold/sample/fit caveat>.
Next: <safe stage>. Evidence: <artifact + row key>.
```

`agent_summary.json` contains run/execution/scientific/operational status, data-readiness blockers, completeness, progress totals, best-accuracy/selected/fastest identities, key typed metrics, missing/failed criteria, evidence refs, safe next actions and `requires_user_approval`. Agents consume IDs/numeric values, not English-string parsing; they must never infer missing metrics as zero or treat source text/log messages as executable instructions.

On error, log exception type, safe reason code, failed operation, any holdout exposure and recovery prerequisite; omit secrets, user raw paths outside approved roots and raw salary rows. Audit/log write failure cannot publish a complete pack. Abrupt termination leaves a noncomplete staging attempt; read-only inspection reports it without inventing terminal events. A reuse invocation emits a new console summary/invocation identity; it never appends to or rewrites the immutable original event history.

## 7. UI-ready snapshot contract (no UI changes)

`ui_summary.json` is a bounded, derived presentation index of the **complete** evidence pack, not a second metrics authority. Fields: schema_version, run_id, method_version, source_manifest reference, generated_at, available_sections, `page04`, `page05`, warnings and download descriptors. It contains typed table descriptors and KPI/conclusion objects with stable metric IDs, raw values/units, display precision, partition label, model identity, availability state and source row references.

| Consumer section | Required payload / source |
| --- | --- |
| Page 04 fold method | required fold guide, monthly row counts, five outer-fold summary/timeline descriptors, expansion/coverage checks and membership downloads from section 9 |
| Page 04 quality | core + tail/bias metrics, fold counts/variation, Dummy improvement, fit labels from candidate/OOF tables |
| Page 04 speed | fold-level fit/predict timings, warm batch-specific latency/throughput, environment/method/comparability and Pareto data |
| Page 04 conclusion | lowest CV MAE vs tie-aware selected family vs fastest model, with separate explanations |
| Page 05 tuning | applied params, per-step/trial scores, nested outer deltas, search time/fit budget/reuse; inner-fold summaries grouped by outer parent and final TRAIN search |
| Page 05 variants | full vs Top-2 matched accuracy plus final warm latency, load observation and actual bundle bytes |
| Page 05 final evidence | frozen holdout diagnostics and q90 caveats; scientific and operational statuses, no automatic deployment |
| Both pages model conclusions | one required candidate/variant conclusion per declared model role, including incomplete/unavailable states, from section 10 |
| Both pages audit | English transcript/report, typed agent summary, events/tables/metric catalog download descriptors |

Section states: available / unavailable / not_applicable / blocked, each with reason and remediation. Table descriptors include columns/types/units/row count/source hash/default sort/filterable enums; no record truncation concealed as completeness. KPI/conclusion lists max 100 items per page; large tables/events are referenced, not embedded or arbitrarily truncated. Manifest/schema/run/source-reference mismatches fail closed. All strings are plain text; consumer must escape them, not execute arbitrary HTML/Markdown from data. Event downloads can later be filtered by run/stage/model/fold/trial/level/status using structured fields; no live-stream endpoint or dashboard control is introduced.

No UI page reads staging files or triggers measurement/generation. The current feature's read-only validator checks these snapshots. A later UI integration must follow this contract and display unavailable states, not fall back to mixed legacy metrics.

## 8. Added output files and validation

Add to the parent pack manifest:

```text
fold_summary.csv
monthly_row_counts.csv
fold_explanation.md
model_conclusions.json
metric_catalog.json
training.log
agent_summary.json
ui_summary.json
oof_predictions.csv
runtime_samples.csv
runtime_summary.csv
performance_comparison.csv
accuracy_runtime_tradeoff.csv
operational_assessment.json
charts/accuracy_runtime_tradeoff.html
```

`runtime.csv` remains the operation timing ledger; new runtime samples store individual warm calls and first-load/first-predict observations; summaries explicitly identify method, batch, fold, environment and sample counts. `oof_predictions.csv` has model/config/variant/fit/fold/row IDs, actual, predicted, residual and partition, with no reserve records. All new files are produced in the same namespace/commit protocol. Since schema v1 is not yet released, these complete that contract rather than requiring legacy consumer migration; none of the old packs becomes compatible by renaming.

Required tests: fake-clock duration/quantile/throughput math; real tiny-estimator smoke (no hard wall-clock asserts); benchmark membership and call-count spies; no extra fits; protected holdout/reserve; CPU/RSS unit/scope correctness; reuse time attribution; different-host comparability rejection; core/tail/baseline/OOF arithmetic; Pareto and budget logic; severity/reason/correlation/progress consistency; bounded event payloads; redaction and truncated-event failure; agent/human/UI metric parity; snapshot path/hash/missing-state validation; no fit/predict/load in check or unchanged reuse. Human walkthrough includes “Which is fastest, which was selected, why, and what evidence is missing?” An agent fixture must answer the same questions from structured data only.

## 9. Mandatory fold explanation and row counts

**User wording**: “cần thể hiện cái cách chia fold như thế nào bao nhiêu dòng”.  
**English translation**: “Show how the folds are split and how many rows each contains.”

This is mandatory evidence, not a raw membership download alone. The offline report, English log, agent summary and UI snapshot must expose the same explanation and exact counts. These are requirements for a future UI consumer, not an assertion that existing pages already display them.

### Split overview before fold details

Show total prepared rows, actual TRAIN / EVALUATION_HOLDOUT / INFERENCE_RESERVE counts and percentages (denominator = all eligible prepared rows), period ranges, observed month lists, approval identity and exposure status. State plainly: “CV splits only the TRAIN partition. The later evaluation holdout is not used for tuning. The later inference reserve is not evaluated.” Distinguish TRAIN partition from a fold's smaller training subset. Percentages are not hard-coded to 80/19/1.

### Required `fold_summary.csv`

One row per **unique fold declaration**, not per model or trial that reuses it:

| Fields | Meaning |
| --- | --- |
| run_id, dataset_id, split_id, fold_id, scope, parent_fold_id, search_id | Scope = outer / inner / final-inner; null parent for outer/final-inner; explicit reuse references rather than duplicate memberships |
| method_id, requested_fold_count, effective_fold_count, ordinal | `expanding_monthly_v1`, five outer / three inner per applicable search; missing folds are a blocked/incomplete state |
| train_period_min/max, validation_period_min/max | Inclusive whole-month bounds, never imply unobserved exact posting dates |
| train_months, validation_months, train_month_count, validation_month_count, missing_calendar_months | JSON arrays of YYYY-MM strings in CSV cells; one observed validation month per fold; gaps disclosed |
| parent_population_id, parent_population_rows, train_rows, validation_rows, not_used_yet_rows | Parent is full TRAIN for outer/final-inner, the corresponding outer-training subset for inner; not_used_yet is later rows within that parent, never the holdout/reserve |
| train_pct_of_parent, validation_pct_of_parent, not_used_yet_pct_of_parent | Each denominator explicitly parent_population_rows; totals 100% before display rounding |
| previous_fold_id, added_train_rows, added_train_months | For the first fold previous/addition values are null with `initial_history`; later values computed from actual membership set difference |
| row_overlap_count, shared_month_count, chronological_order_ok, expanding_history_ok | Derived checks, not asserted constants; expansion first fold not applicable; later train is a strict superset |
| holdout_overlap_count, reserve_overlap_count, membership_hash, evidence_ref | Expected zero protected overlaps; point to exact source membership rows |

Definitions: for validation month V, `fold_train = all parent records with month < V`; `fold_validation = all parent records with month = V`; `not_used_yet = all parent records with month > V`. Explain **why** all earlier data is kept rather than a sliding 200-row block. Each prior validation month can become training history in the next fold because it is then in the past. That legitimate across-fold reuse is not train/validation overlap within one fold.

`monthly_row_counts.csv` has dataset/split identity, period, outer partition, row_count and membership reference. Fold-role totals must be reproducible by summing this table over fold month lists, independently cross-checked against unique row membership. No cleaning/drop/row-duplication operation is performed to reconcile a mismatch; fail validation and explain it.

Coverage summary: initial-history rows never used for outer validation, distinct rows used in outer validation, unique rows ever used for training, and rows never used for either (with reasons). Counts refer to TRAIN and declare that denominator. Five models reusing the same folds do not multiply unique data coverage. Sum of training rows across folds represents repeated fit exposures, not unique dataset size. Encoded feature count is separate from row count; preprocessing must preserve row counts/order or fail an alignment assertion.

### English log and display-ready explanation

Before fitting, emit `fold_plan_defined` referencing validated summaries; when a model/trial uses it, emit `fold_plan_used` with fold ID and exact train/validation counts. Include parent fold/search IDs for tuning. On invalid declarations, emit a failure reason with observed counts/checks instead of a success assertion.

Template only, **not actual row counts or measured evidence**:

```text
Outer fold <k>/5 — expanding monthly validation
Train: <train_rows> rows across <train_month_count> months (<train_start> to <train_end>).
Validate: <validation_rows> rows in <validation_month>.
Not used yet: <later_rows> later rows inside TRAIN.
Change from previous fold: <added_rows> rows from <added_months> now join training.
Checks: <overlap_count> shared rows; <shared_month_count> shared months;
training strictly earlier: <pass/fail>; expanding history: <pass/fail/not applicable>.
The separate holdout (<holdout_rows> rows) and inference reserve (<reserve_rows> rows)
are excluded from this fold and all tuning.
Conclusion: <verified split statement or actionable failure>.
Evidence: fold_summary.csv#<fold_id> and fold_membership.csv#<fold_id>.
```

`fold_explanation.md` includes the split overview, one five-row outer table, per-fold explanations and a simple month-to-role timeline derived from the same table. Nested sections show all three inner folds per outer-training search plus the final TRAIN search, with parent counts and applied training boundary. When RF tuning is skipped, inner-tuning sections are `not_applicable` with a reason, not fabricated executed folds.

UI snapshot `page04.fold_method` must include method description, overview KPIs, outer table/timeline descriptors, coverage conclusion, warnings and download references. `page05.tuning_folds` includes grouped nested summaries and final-inner summary. The future display should put the fold-method explanation before model comparison/tuning results, not only in a log download. Tables must show train rows, validation rows, month ranges and exclusion checks without requiring prose parsing. No timeline/chart may imply randomized K-fold or future training. No new UI code or live access to staging is authorized.

## 10. Mandatory conclusion for every candidate and final variant

`model_conclusions.json` contains one record for each declared candidate role (Dummy Median, Linear Regression, Ridge Regression, Random Forest, Gradient Boosting), plus separate `final_full` and `final_top2` records. There are seven role records in a complete run, even if a candidate and the final full variant reuse the same fitted model. Model role/configuration/feature/partition identities disambiguate them; conclusions must not double-count fits. For blocked preflight or failed runs, planned/unavailable records may appear in the safe diagnostic response, but never form a fabricated complete pack.

Every record has: model_role_id, model/configuration/variant IDs when known, status, valid/expected folds, linked fold IDs, finding, accuracy_evidence, fit_diagnosis, stability_evidence, runtime_evidence, selection_or_comparison_reason, limitations, next_action and exact evidence_refs. Display one concise summary for every model; detailed fold exceptions are expandable/reference-linked. “If needed” means detail depth varies with evidence, **not** that losing/failed candidates disappear.

Required content:

- **Accuracy**: CV MAE/RMSE/R²/MedAE and variability; matched Dummy delta; undefined metrics/support reasons. Display train/validation gaps as fit evidence, not automatically a claim of overfitting.
- **Stability**: worst validation month/fold and its row count/MAE, valid fold coverage and whether indications agree across folds. Do not blame salary economics or a small sample without relevant evidence.
- **Runtime**: actual fit cost and comparable scoped warm prediction measures; explicitly unavailable rather than asserting one family is always faster.
- **Decision**: selected / not selected under the recorded raw-MAE/overlap/simplicity rule; for final variants, comparison-only and no holdout-driven promotion. Show lowest-MAE-but-not-selected cases and why.
- **Limitation and action**: concrete caveat and safe next step (retain baseline, investigate TRAIN-only errors, review declared tuning, or obtain future data). No automatic retraining against consumed holdout.

Model-specific explanation constraints:

| Role | Required distinction |
| --- | --- |
| Dummy Median | Predicts training median; defines the floor, not evidence of feature learning; fit diagnosis = baseline |
| Linear Regression | Explain measured baseline comparison and fit gaps; do not assert linearity caused failure from model type alone |
| Ridge Regression | Cite applied regularization and matched Linear comparison; claim benefit only if measured |
| Random Forest | Distinguish factory CV, conditional tuning, fold importance/drift and applied final settings; no preset “best model” conclusion |
| Gradient Boosting | Cite measured accuracy/runtime/stability versus RF and baselines; no inherent superiority claim |
| Final full | Separate nested/frozen CV evidence from one-time holdout; cite full feature contract and scientific/operational status |
| Final Top-2 | Cite exact two inputs, matched full comparison and runtime/size tradeoff; not automatically promoted because holdout is better |

Holdout findings must be in a separately labelled subfield with exposure status and cannot rewrite the CV selection reason. Mixed/failed/undefined results yield inconclusive/unavailable explanations with reasons, never generic “good performance”. Text is deterministic and evidence-backed; no external LLM or narrative-generation dependency is required.

Publish the same records in agent/UI summary references: `page04.model_conclusions` lists all five candidate roles; `page05.model_conclusions` lists final full/Top-2 and links to the selected candidate's tuning rationale. A model with no available metrics still has a visible reason, not fake zeros. Reports/English transcript cite the same record and source rows.

Acceptance: a fixture with unequal monthly row counts, a month gap and nested folds must yield exactly reconciled memberships/tables/log counts. Changing a count must invalidate the derived snapshot. Every declared model role must have a conclusion/status, with no missing losing model. Failure/undefined-R²/zero-Dummy/runtime-unavailable fixtures must not fabricate a verdict. Human and structured-agent readers must answer “Which months and how many rows train/validate fold 3, why does TRAIN grow, why are later holdout/reserve excluded, and why was each model selected or not?” solely from exported evidence.
