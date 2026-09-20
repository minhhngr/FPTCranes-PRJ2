# Feature Specification: Training-Only Split and Validation Evidence

**Feature Branch**: `006-validate-training-split`  
**Created**: 2026-09-19  
**Status**: Implementation approved and fixture-verified; real-data and two-reader acceptance remain gated  
**Input**: User request: validate training only; approximately 80% training / 19% evaluation holdout / 1% future inference-only reserve, using strictly separated whole months; five expanding monthly folds; five-model comparison, bounded tuning, explainability and detailed English logs; no UI, preprocessing or data-preparation changes.

## User Scenarios & Testing *(mandatory)*

### User Story 1 — Understand and verify every data boundary (Priority: P1)

As a reviewer, I can read an English report explaining where each prepared record belongs, how each temporal fold works, and why no future or reserved data enters fitting or selection.

**Why this priority**: All later metrics depend on a correct and auditable split.

**Independent Test**: A small dated dataset produces a reproducible membership report with disjoint partitions and five genuinely expanding monthly folds; invalid boundaries prevent training.

**Acceptance Scenarios**:

1. **Given** existing cleaned, unencoded project data and an approved split policy, **when** training is planned, **then** the report states requested percentages, actual counts and percentages, rounding, ordering, date precision, period boundaries and every record's membership.
2. **Given** five eligible validation months within the training/development partition, **when** folds are constructed, **then** each validation set is one whole month, all training months precede it, and each successive training set contains the preceding training set and all newly available earlier records.
3. **Given** fewer than six eligible months, an empty partition, missing dates or an invalid schema, **when** validation runs, **then** it fails with an English explanation rather than substituting random or sliding-block CV.
4. **Given** the Page 06 reserve, **when** training, tuning, ablation or evaluation runs, **then** those records and their targets are excluded from all such operations. Target-independent membership bookkeeping is permitted; reserve inference and UI changes are not part of this feature.
5. **Given** existing preprocessing definitions, **when** a fold is fitted, **then** the same definitions are reused with fresh training-only fitted state; globally encoded prepared views are not used as fold-training input.
6. **Given** an outer or nested fold, **when** a reader opens the exported fold explanation, **then** they can identify its training/validation month lists and exact row counts, parent population, rows added since the previous fold, later unused rows and verified overlap/chronology checks. These facts must be available in the UI-ready table, not only raw membership downloads.

### User Story 2 — Compare five model families using readable evidence (Priority: P1)

As a reviewer of the evidence intended for Page 04, I can compare quality first, inspect runtime second, and understand the selected family without opening source code.

**Why this priority**: Model choice must be justified by comparable measurements rather than assumed to be Random Forest.

**Independent Test**: All five families produce train and validation metrics on identical fold memberships, with a traceable ranking and an English conclusion.

**Acceptance Scenarios**:

1. **Given** validated folds and a frozen feature policy, **when** comparison runs, **then** Dummy Median, Linear Regression, Ridge, Random Forest and Gradient Boosting each report fold-level MAE, RMSE, R² and MedAE, paired training metrics, aggregate variation and measured runtime.
2. **Given** completed results, **when** the report explains the decision, **then** it discusses R² and MAE first, runtime next, and finally the selection rule and winner; MAE governs selection under the project constitution.
3. **Given** overlapping candidate performance, **when** the winner is chosen, **then** a predeclared variance-aware simplicity rule is applied and its evidence is logged; runtime cannot silently override the selection policy.
4. **Given** paired train/validation evidence, **when** fit quality is discussed, **then** underfitting, good fit and overfitting are explicitly qualified diagnostic indications with cited values and predeclared rules. Insufficient evidence is reported as inconclusive rather than forced into a label.
5. **Given** feature-family ablation and fold importance results, **when** the report is read, **then** the exact removed families, baseline, matched folds, error changes and importance drift are visible; no causal claim is made.
6. **Given** comparable hardware, threads, folds and benchmark workloads, **when** runtime is reported, **then** readers can distinguish pipeline fit cost, ordinary scoring, warm single/batch prediction latency, throughput and measurement overhead, with raw samples and sample counts.
7. **Given** different winners for lowest MAE and fastest runtime, **when** the conclusion is generated, **then** both identities and the accuracy/speed tradeoff are shown without silently overriding the MAE/simplicity selection policy.
8. **Given** all five candidate roles, **when** conclusions are exported, **then** each model—including non-selected or failed models—has its own evidence-backed accuracy, fit, stability, runtime, decision, limitation and next-action explanation. Final full and Top-2 variants receive separate scoped conclusions.

### User Story 3 — Tune and assess frozen models honestly (Priority: P2)

As a reviewer of the evidence intended for Page 05, I can follow the best-model tuning steps, compare the full feature set with a declared two-feature variant, and distinguish development validation from final holdout evidence.

**Why this priority**: Final reporting must not reuse holdout feedback to optimize the model.

**Independent Test**: Controlled fixtures prove tuning stays inside development training boundaries and final evaluation cannot run until all configurations and feature variants are frozen.

**Acceptance Scenarios**:

1. **Given** the family-selection decision, **when** Random Forest is selected, **then** a bounded, logged, development-only nested temporal tuning procedure runs. If another family wins, RF tuning is skipped with a reason, not used to force RF into the winner position.
2. **Given** the final configuration and two-feature definition, **when** variant comparison runs, **then** full and two-feature variants use identical evaluation memberships and report their actual fitted settings; no holdout-driven feature selection is allowed.
3. **Given** the frozen evaluation declaration, **when** final evaluation is authorized, **then** all required candidate diagnostics and the frozen full/two-feature comparisons are produced in one evaluation phase, without changing the pre-evaluation winner.
4. **Given** a historically evaluated dataset, **when** evidence is published, **then** it is labelled historical/re-evaluation evidence, never a new pristine locked test. A genuinely untouched future evaluation remains unavailable until appropriate data exists.
5. **Given** final predictions, **when** diagnostics are generated, **then** actual-versus-predicted records, residuals, subgroup error slices, raw permutation importance, encoded estimator importance and an empirical q90 absolute-error band are available with scope and caveats.
6. **Given** a failed quality gate, **when** the workflow concludes, **then** it recommends a development-only revision or new future evaluation data. It does not repeatedly optimize against the same exposed holdout.

### User Story 4 — Follow the training story without reading code (Priority: P2)

As a non-specialist or a future UI consumer, I can read structured and plain-English logs explaining each stage, its evidence, its conclusion and the next action.

**Why this priority**: Training results must be understandable and consumable independently of a UI redesign.

**Independent Test**: A complete fixture run and a failed fixture run each produce ordered, traceable English reports; a reader can reconstruct the split, selection and outcome solely from the saved evidence.

**Acceptance Scenarios**:

1. **Given** any stage, fold or tuning trial, **when** it starts, completes, fails or is reused, **then** the log includes its identity, inputs, parameters, progress, elapsed time, outcome and linked evidence.
2. **Given** a finished report, **when** a novice reads it, **then** they can answer Who, What, When, Where, Why and How, identify the selected model and explain the reserve restriction.
3. **Given** the publication manifest, **when** an offline consumer loads it, **then** model comparison and best-model evidence can be parsed without console scraping or executing training. Existing Pages 04–06 remain unchanged.
4. **Given** incomplete, incompatible or stale artifacts, **when** a report is requested, **then** the status is explicit and actionable; no fabricated numbers or silent training occurs.
5. **Given** one validated run, **when** a human reads its English transcript, an agent reads its structured summary and a future UI reads its snapshot, **then** their metrics, model identities, partition labels, decisions and evidence references agree exactly before display rounding.
6. **Given** a failed, skipped or reused stage, **when** its events are consumed, **then** stable reason codes, operation/fold/trial IDs, honest progress counts and safe next actions explain the outcome without requiring prose parsing or any training action.

### Edge Cases

- Whole-month chronology takes priority over exact row percentages. Report actual ratios and deviations; never split a month to improve the percentages.
- The current final month alone is about 20% of the snapshot. A near-1% strictly later reserve may be unavailable; report the actual proposal rather than implying that 80/19/1 was achieved.
- Known historically exposed rows cannot form an unseen inference reserve. Missing or unverifiable fresh reserve provenance blocks a complete compliant run; preflight can still report the reason without fitting.
- Too few months for five outer folds or for nested inner folds must block the affected evaluation, not silently downgrade the method.
- A validation month with one row or constant targets may make R² undefined; report null with a reason, never an invented zero or infinity.
- Unknown categorical values must follow the existing preprocessing contract, fitted separately inside each fold.
- Duplicate records, inconsistent prepared-source identities, missing/nonfinite targets and prohibited feature columns must be diagnosed without silently recleaning data.
- Failed candidates/trials and cancellation must remain visible; incomplete comparison is not an approved winner.
- Importance features absent from a fold's fitted vocabulary must be explicitly distinguished from observed zero importance.
- Small subgroup counts, zero baseline error and unstable train/validation ratios require explicit limitations.
- Invalid workspace paths, run IDs, feature lists, seeds, search bounds or overwrite requests fail closed before fitting or publication.

## Requirements *(mandatory)*

### Scope and workflow

In scope: training-side feature-policy validation, temporal partition membership, model fitting, comparison, ablation, nested tuning, evaluation, diagnostics, provenance and English evidence publication.

Out of scope: changes to data cleaning, upstream data preparation, preprocessing algorithms, source data, Streamlit pages, Page 06 prediction behavior, deployment infrastructure, live monitoring services and automatic retraining schedulers.

The lifecycle is **Features → Train → Evaluate → Tune → Deploy → Monitor → Retrain**. In this feature, Deploy/Monitor/Retrain are documented recommendations and handoff statuses only. “Good: deploy model” means eligible for human deployment review, not an automatic deployment or a production-fitness claim. “Bad: loop” means revise using development evidence; it never authorizes repeated holdout-based optimization.

### Functional Requirements

- **FR-001**: Reuse existing project-prepared, cleaned raw features and existing preprocessing definitions without changing cleaning/preparation behavior. Resolve the user's `output/` reference to the actual repository `outputs/` directory and document exact inputs.
- **FR-002**: Declare the target `annual_salary_usd`, allowed features, blocked target-derived fields, partition policy and exposure history before target-aware training analysis.
- **FR-003**: Target approximately 80% training, 19% final validation/evaluation holdout and 1% future inference-only reserve. Assign contiguous whole-month populations with `max(training month) < min(holdout month)` and `max(holdout month) < min(reserve month)`. Chronology overrides exact percentages. Report and explicitly freeze actual boundaries/counts/ratios before fitting. All tuning and selection use CV only within training; the 19% is opened only after decisions are frozen.
- **FR-004**: Exclude reserved records from preprocessing fitting, feature selection, training, CV, tuning, importance, uncertainty estimation and model-quality evaluation. Reserve is for later inference only, not a test set. This feature records target-free membership and provenance but neither predicts nor exports reserve targets. Known previously used records cannot be called unseen; unknown/invalid exposure provenance blocks a complete compliant run instead of silently relabelling historical data.
- **FR-005**: Use exactly five expanding monthly outer folds within the approved training/development population. Publish all memberships, date bounds, row counts, omitted/initial-history months and assertions proving strict monthly ordering, no overlap and expanding history.
- **FR-006**: Fit existing preprocessing separately on each applicable fold-training population; never reuse full-development fitted encoders/scalers in CV.
- **FR-007**: Evaluate the five required families on identical folds and features, with train/validation MAE, RMSE, R², MedAE and their units/directions explained.
- **FR-008**: Report fold metrics, explicitly defined aggregates and variation, measured preprocessing/fit/prediction or clearly labelled combined timings, hardware/environment and seed. Do not claim statistical significance from five fold means alone.
- **FR-009**: Rank families by lowest mean temporal CV MAE and apply a predeclared variance-aware simplicity rule, while presenting R²/MAE analysis followed by runtime analysis and the final decision. Freeze the family before bounded RF tuning.
- **FR-010**: Publish paired fit diagnostics, feature-family ablation, fold-level feature importance and importance-drift evidence with declared alignment and measurement rules.
- **FR-011**: Keep the familiar stepwise RF search narrative, but make fresh tuning conform to nested temporal development-only evaluation and MAE-based selection. Log all search bounds, carried parameters, trials, inner/outer fold identities, seeds, scores and actual applied final parameters. Existing non-nested R²-ranked results cannot be relabelled compliant.
- **FR-012**: Compare the full feature set with a predeclared two-feature variant using matched populations; do not select those features from holdout importance. The plan proposes `job_category` and `years_of_experience`, matching the existing supplemental two-input experiment; this assumption is included in task review. Neither variant may be refitted on the 19% or the reserve.
- **FR-013**: Freeze family/configuration/feature variants and evaluation policy before opening final holdout targets. Evaluate required candidate diagnostics plus final variants together without using results to change the frozen winner. Record access history and prevent an unchanged rerun from reopening/recomputing final evaluation unnecessarily.
- **FR-014**: Publish prediction/residual evidence with an explicit sign convention, subgroup counts and metrics, raw permutation importance with declared scoring/repeats, and encoded estimator importance with method labels. Unsupported estimator-specific importance is unavailable, not manufactured.
- **FR-015**: Publish the empirical 90th percentile of absolute errors with its estimation population, quantile method and interval rule. Same-population coverage is descriptive, not an independently calibrated 90% guarantee. Every prediction summary includes this qualified band or MedAE.
- **FR-016**: Keep modeling as continuous salary regression. Explain binary/multiclass metrics only in educational documentation labelled not applicable to the current model. Do not train classifiers or invent labels, salary thresholds or classification scores. Future classifier work requires separately approved targets and class definitions.
- **FR-017**: Emit ordered, detailed machine-readable events plus a plain-English step-by-step report, metric glossary and evidence-backed conclusions for each stage and the overall run.
- **FR-018**: Log Who/What/When/Where/Why/How, requested and actual splits, data/configuration/producer/dependency identities, features, blocked fields, fold/trial counts, fitted parameters, measured durations, failures, reuse decisions and artifact locations. Do not log credentials or claim raw data dumps are necessary for completeness.
- **FR-019**: Publish explicit Good/Bad/Blocked/Inconclusive outcome and next action, based on documented pre-run acceptance criteria and limitations. Unknown production thresholds cannot default to Good. Runtime failures are not evidence of underfitting.
- **FR-020**: Define versioned producer/consumer contracts for evidence intended for Pages 04–05, without modifying any UI consumer in this feature. New split semantics must not overwrite old packs under the same identity or silently activate a new serving model.
- **FR-021**: Reuse unchanged valid artifacts; when approved training logic, splits, schema or dependencies change, regenerate or explicitly invalidate affected results before reliance. Preserve historical artifacts as historical, including test-exposure provenance.
- **FR-022**: Validate every training-controlled path, run ID, configuration, feature list and dataset schema before use; document safe errors, containment, bounds and publication-failure behavior.
- **FR-023**: Document affected training functions/stages, validation rules, artifacts and scientific meaning. Generate and reference offline model-comparison, final-diagnostic and importance charts alongside authoritative numeric evidence.

- **FR-024**: Define and record model performance as separate accuracy, temporal stability, fit cost, inference latency/throughput and artifact-size evidence. Identify best accuracy, selected family, fastest model and accuracy/fit-cost Pareto frontier separately; no invented weighted optimum or runtime-driven override.
- **FR-025**: Measure operation wall/CPU time with explicit boundaries and runtime context. Include fold/search/run cost, actual/reused/failed fit counts, measurement overhead, serialization/load observations and process-scoped memory when supported. Do not confuse inclusive child timings with independent elapsed time or process high-water memory with per-model memory.
- **FR-026**: Benchmark prediction with a bounded, reproducible protocol using only already fitted pipelines and TRAIN features: single and bounded batch inputs, warmups, repeated samples, p50/p90/p95 latency and correctly scoped throughput. No benchmark fits, holdout/reserve access or automatic rebenchmark on reuse; no claim of UI/production latency.
- **FR-027**: Extend regression evidence with signed bias, tail absolute error, matched Dummy-relative improvement, paired train/validation gaps, worst-fold context and tuning/Top-2 deltas; define units, formulas, aggregation and undefined cases in a machine-readable metric catalog. Preserve MAE as the primary ranking metric.
- **FR-028**: Generate a plain-English chronological transcript, narrative report and structured agent summary from the same validated evidence. Logs must include severity, stable event/reason codes, invocation/operation/fold/trial correlation, bounded progress, evidence references, failures, reuse and approval-aware next actions. No secrets or raw data dumps.
- **FR-029**: Export a validated, bounded UI-ready snapshot for Pages 04–05 with typed metrics/table descriptors, model/partition identities, source references, units, precision, availability states, warnings and download metadata. This does not implement pages, live streaming or training buttons.
- **FR-030**: Evaluate optional predeclared operational budgets independently from scientific outcome. Missing budgets mean not assessed, not passed; missing configured measurements mean unavailable. Failed operational budgets block a deployment-review recommendation but do not change the scientific winner or authorize another holdout evaluation.
- **FR-031**: Verify metric parity across human/agent/UI representations, benchmark arithmetic and workload comparability, no extra fits or protected-population access, honest timing/reuse attribution, actionable errors and fail-closed artifact validation before publication.

- **FR-032**: Provide a mandatory English fold walkthrough and UI-ready summary containing actual train/validation rows, month lists/ranges, parent counts/percentage denominators, newly added history, later unused rows, excluded holdout/reserve counts and verified leakage/expansion checks. Include five outer folds and parent-labelled nested tuning folds; reconcile counts against monthly totals and membership, not hard-coded examples.
- **FR-033**: Publish a conclusion/status for every candidate model and each final full/Top-2 variant, not only the winner. Bind accuracy, fit/stability and runtime evidence to the correct configuration/population; explain selection/non-selection, limitations and safe next actions. Missing/failed evidence yields a visible reason rather than an invented favorable conclusion.

### Constitutional Requirements *(mandatory)*

- **Data boundary**: Regression target and existing feature exclusions remain authoritative. Monthly CV is training-only, strict and expanding. Upstream preparation is unchanged; training may create new membership manifests over its outputs. The 19% is a final evaluation holdout, never a tuning population. Whole-month ordering is strict across all three partitions; actual shares may differ. Fresh reserve provenance is an execution gate, not permission to modify upstream data.
- **Artifact contract**: Existing `outputs/04_model_comparison/`, `outputs/05_best_model/`, `outputs/ui_evidence/` and saved bundles are historical reference inputs, not automatically valid evidence for a new split. The new versioned training-evidence contract is defined in `contracts/training-evidence.md`; existing consumers must not silently receive changed meanings.
- **Streamlit boundary**: No Streamlit code changes or training actions. “Can be logged on the UI” means offline consumable evidence, not wiring new displays. UI integration is a separate approval.
- **User input validation**: CLI/configuration paths, names, ranges, feature enums, finite values, data schema and period constraints require fail-closed validation. Uploads, widgets, credentials and prediction input behavior are unchanged and outside this feature.
- **Scientific interpretation**: Importance is model reliance, not causation. This academic snapshot does not establish production fitness. Historical target exposure cannot be erased by renaming or repartitioning. Binary/multiclass coverage is educational only; new classifier training is outside scope.
- **Documentation impact**: English split/fold guide, feature policy, tuning steps, metric glossary, runtime method, per-stage conclusions, final recommendation, artifact map and limitations.
- **Verification evidence**: Failing-first membership/leakage tests; fold and tuning spies; metric reference calculations; chronology/rounding edge cases; artifact-contract and no-UI-mutation tests; bounded offline end-to-end validation only after approval.
- **Graphify/Karpathy review**: Existing graph query identified `temporal_cv_splits`, `tune_random_forest`, training-audit and supplemental producers; truncated graph results were followed by direct source inspection. Prefer minimal training-only changes after scope resolution; no generic training platform or upstream rewrite.

### Key Entities

- **Dataset identity**: Prepared source, content identity, feature contract and historical exposure.
- **Partition declaration**: Approved allocation, chronology, record membership, rounding and reserve restrictions.
- **Fold declaration**: Training/validation months and membership, outer/inner purpose and leakage assertions.
- **Experiment/trial**: Family, feature variant, configuration, seed, fit scope and runtime.
- **Metric/diagnostic evidence**: Value, unit, partition, fold, method and undefined-value reason.
- **Selection/evaluation lock**: Frozen decisions, acceptance rules, exposure ledger and evaluation authorization.
- **Training report**: Ordered events, 5W1H context, conclusions, status and machine-consumable evidence references.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Every eligible record is assigned exactly once to an approved outer partition; reserve overlap with every training/evaluation population is zero.
- **SC-002**: Exactly five valid monthly outer folds have strictly earlier training months and nested expanding training memberships; unsupported input fails before fitting.
- **SC-003**: A complete comparison contains 25 candidate-fold results, each with paired train/validation MAE, RMSE, R² and MedAE (or explicit undefined reasons), runtime and traceable membership/configuration identities.
- **SC-004**: Every selection, tuning and feature-variant decision can be reconstructed from development evidence; zero holdout or reserve targets influence those decisions.
- **SC-005**: Every stage and trial has a terminal completion/failure/cancellation/reuse event; every reported conclusion cites saved evidence and its evaluation scope.
- **SC-006**: Two independent readers using only the English report can correctly identify the split, reserve restriction, meaning/direction of MAE and R², selected family, tuning outcome and next action. This is a planned acceptance walkthrough, not a claimed completed test.
- **SC-007**: Source data, upstream preprocessing/preparation files, existing UI code and existing serving artifacts remain unchanged during implementation; new evidence is clearly distinguished from legacy evidence.
- **SC-008**: An unchanged valid rerun reuses published evidence without fitting; a changed split/producer contract cannot masquerade as a compatible old run.
- **SC-009**: Each required benchmark group contains 30 raw measured calls after three warmups, or an explicit failure/unavailability reason; all aggregates are reproducible from samples. Additional benchmark work is bounded to 1,784 predict calls and zero fits.
- **SC-010**: Every reported KPI and conclusion in human, agent and UI-ready exports resolves to matching run/model/partition evidence and agrees before rounding; corrupted references or mixed run IDs fail validation.
- **SC-011**: A person and a structured-data-only agent can identify best accuracy, selected model, fastest comparable model, runtime cost, remaining blockers and safe next action without source-code inspection or retraining.
- **SC-012**: Runtime evidence differentiates overlapping timings, unequal fold workloads, benchmark overhead, historical reuse and unconfigured budgets; no unavailable measurement becomes zero or a passing readiness claim.

- **SC-013**: Every displayed/logged fold row count and month range reconciles exactly with unique source membership and monthly totals; a fixture with unequal month sizes, gaps and nested folds proves correct expansion and explicit percentage denominators.
- **SC-014**: A complete experiment exposes five candidate conclusions and two final-variant conclusions with source references; human and structured-agent readers can explain fold 3's months/counts and every model's selection/comparison status without code inspection.

## Assumptions

- `output/` means this repository's `outputs/`; actual cleaned raw inputs are under `outputs/01_data_basic_clean/` and `outputs/02_data_ready_for_ml/`.
- “Do not touch preprocessing/data preparation” forbids changing upstream algorithms/artifacts, not fitting fresh copies of the established preprocessing pipeline inside training folds.
- Full-feature selection is restricted to the existing approved feature policy. Feature engineering here means evaluating those established features and family ablations, not adding transformations or target-derived fields.
- The two-feature proposal is `job_category` + `years_of_experience`; it remains a declared assumption for review, not a claim of optimality.
- Deployment, monitoring and retraining are handoffs/recommendations only. No new service or scheduler is requested.
- Existing results may guide understanding of the code, but do not satisfy new expanding-fold or untouched-reserve claims.

## Translated Third-Party Feedback

1. “Check again whether this fold-splitting approach is correct or whether anything is missing.”
2. “First, analyze the chart evaluating the five models using R² and MAE. Then evaluate runtime, and finally choose the best model.”
3. “For the best model, keep the previous step-by-step tuning approach. Then compare it with a two-feature version (run it again and export the results to the app).”

Interpretation: preserve the readable stepwise explanation, not incorrect legacy evaluation semantics. Export backend evidence only; do not edit the app. A new full-versus-two-feature run is justified only when the new approved experiment requires it, not merely for a demo. See FR-011–FR-012 and FR-020.

## Clarification Record — User Decisions (2026-09-19)

1. **19% role**: Validation/evaluation holdout. All tuning/CV occurs only inside the 80% training partition. The remaining 1% is unseen inference/prediction data simulating real-world use, not model evaluation.
2. **Chronology**: Preserve chronological order and whole-month boundaries. Approximate 80/19/1 is allowed; partitions remain strictly temporally separated. Same-month holdout/reserve allocation is not allowed.
3. **Task type**: Regression remains the modeling task for continuous salary. Classification metrics are explained only; no classifiers without separately specified targets/classes.

### Practical limitations carried into planning

The existing snapshot ends in March 2026, which contains 298 of 1,499 rows and has already been evaluated. There is no verified unseen future month in the inspected artifacts. An offline preflight must expose this limitation and block complete compliant execution on this snapshot; it must not silently create a fictitious unseen 1% from March. Acquiring/preparing new future data remains outside scope. Implementation can be validated on isolated chronological fixtures and can run on valid future prepared inputs once provided through the existing preparation process.

The 19% plays the constitutional locked-test role despite its user-facing name “evaluation holdout”. Historically exposed holdout evidence must retain that limitation; it cannot establish production readiness. One-time evaluation means one frozen evaluation transaction per holdout identity, not a fresh lock obtained merely by changing a run ID.

## Fold-Explanation Review Amendment

“cần thể hiện cái cách chia fold như thế nào bao nhiêu dòng” means “Show how the folds are split and how many rows each contains.” Existing membership/count requirements were present, but the visible walkthrough and per-model conclusions were insufficiently explicit. FR-032–033 and contract sections 9–10 make them mandatory in the English/agent/UI-ready evidence. Actual UI implementation remains a separate approval; no fold row counts are invented before a verified run.

## Runtime and Readability Review Amendment

The user requested more detailed runtime/performance metrics, logs readable by humans and agents, and evidence that can be shown in the UI. [contracts/runtime-observability.md](contracts/runtime-observability.md) specifies the measurement protocol, metric dictionary, accuracy/speed distinctions, English/structured outputs and future UI snapshot contract. “Best performance” does not assert a fastest or most accurate model before measurement. No model was benchmarked during this review. The amendment stays offline and does not expand into UI implementation.

**Review gate disposition**: The user explicitly approved the amended tasks before implementation. That approval did not authorize fabricating future data, weakening exposure checks, changing the UI/preparation pipeline, or running the blocked current dataset.
