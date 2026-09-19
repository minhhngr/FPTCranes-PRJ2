# Feature Specification: Evidence-Backed Model Comparison, Diagnostics and Prediction UI

**Feature Branch**: `005-improve-model-ui`  
**Created**: 2026-09-19  
**Status**: Baseline, semantic-color, report-readability and approved table/training-explanation amendments implemented and automatically verified; human/browser acceptance remains pending  
**Input**: [Stakeholder UI intent](../../docs/spec-imporve-ui.md): improve Pages 04, 05 and 06 using visual-first layouts, directly labelled charts, concise evidence and controlled salary scenarios. User clarification on 2026-09-19 expands scope to generate missing evidence offline in new Python files, not the existing god module; existing main-training behavior and calculations remain unchanged. Post-implementation feedback first required stronger semantic color, red test-error columns, and priority-based chart widths; a second review reports overlapping labels, unreadable numbers and insufficient detail/conclusions for third-party readers. A third review requests consistent bold/italic table emphasis and high-priority collapsible explanations of temporal validation and hyperparameter tuning, backed by existing evidence and downloadable audit logs where compatible.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Compare model families using traceable evidence (Priority: P1)

As an executive reviewer, I want to compare candidate families and their chronological validation performance so I can understand the selection and its limitations without reading lengthy narrative.

**Why this priority**: The comparison establishes the evidence and terminology needed to interpret later diagnostics and predictions; it is independently useful without changing inference.

**Independent Test**: Open Page 04 with a verified output pack and reconcile every displayed model, score and fold with its source. Repeat with missing train scores and changed rankings; the page must not manufacture values or keep stale winner labels.

**Acceptance Scenarios**:

1. **Given** a compatible comparison pack, **when** Page 04 opens, **then** a five-chip ribbon reports candidate count, effective fold count, winning family, best mean validation MAE and Dummy Median baseline; four tabs organize overall comparison, Random Forest, Gradient Boosting and baseline controls.
2. **Given** verified train and validation scores, **when** the overall tab is viewed, **then** labelled validation-MAE columns and a training-MAE line precede the executive table, with the actual Dummy Median MAE as a dashed reference; a second chart compares validation/train MAE ratio and validation R² on explicitly distinguished axes.
3. **Given** train scores lack provenance or are unavailable, **when** the tab opens, **then** validation evidence remains visible and train-dependent gaps, ratios and fit assessments explicitly state insufficient evidence rather than using the document's example values.
4. **Given** chronological fold evidence, **when** a deep-dive is opened, **then** the user sees fold-level comparison, feature-reliance stability and concise technical facts sourced from that model/configuration; Random Forest is not labelled winner if the actual ranking changes.
5. **Given** baseline evidence, **when** the controls tab opens, **then** Linear, Ridge and Dummy are contrasted with the leading nonlinear model without asserting a proven matrix-conditioning failure or causal explanation unsupported by evidence.

---

### User Story 2 - Audit the selected configuration and its limitations (Priority: P2)

As a model auditor, I want to see the selected configuration, development validation, test diagnostics, tuning evidence and practical uncertainty together so I can assess what the experiment actually supports.

**Why this priority**: Distinguishing family comparison from final-configuration evaluation prevents misleading generalization claims.

**Independent Test**: Open Page 05 with known metrics, predictions, tuning and importance evidence; reconcile the scoreboard to the final configuration and test population, then remove an optional pack and verify its section explains the gap.

**Acceptance Scenarios**:

1. **Given** a verified final-model pack, **when** Page 05 opens, **then** six chips identify the model, test R², MAE, MedAE, explicitly named CV-to-test R² gap and empirical q90 band; the actual test count and period are visible.
2. **Given** matching final-configuration CV and test metrics, **when** the generalization section is viewed, **then** charts precede the scorecard, show labelled dollar errors and a separately scaled R² comparison, and derive deltas from unrounded values; baseline family CV is not substituted for tuned CV.
3. **Given** test predictions, **when** diagnostics are viewed, **then** actual-versus-predicted scatter includes the identity line and residual distribution includes a zero reference and boxplot; neither shape is described as proof of unbiasedness or absence of overfitting.
4. **Given** recorded tuning results, **when** the tuning section opens, **then** a collapsed methodology explanation describes the actual anchor search and coordinate sweeps; four labelled MAE/R² sensitivity charts precede their evidence tables, and applied parameters are distinguished from sweep winners.
5. **Given** importance evidence, **when** feature reliance is viewed, **then** raw-feature permutation importance with error bars and encoded importance are labelled as distinct measures; a Top-2/full-feature comparison is shown only when compatible, independently traceable evidence exists.
6. **Given** q90 evidence and test errors, **when** the trust section is viewed, **then** empirical coverage, tail ratio and calibration population are disclosed. Three records may be called pristine only if their exclusion from training, tuning and benchmark scoring is proven; ordinary test examples must not be relabelled reserved.

---

### User Story 3 - Build and run controlled salary scenarios (Priority: P1)

As a recruiter using an academic scenario tool, I want constrained inputs, a reviewable queue and transparent prediction bands so that I can compare scenarios without entering unsupported values or mistaking model output for a compensation guarantee.

**Why this priority**: Correct serving identity and input validation are essential safety requirements even though this journey follows diagnostics in the page sequence.

**Independent Test**: With a compatible approved inference bundle, add valid scenarios, reject invalid combinations, load eligible benchmark examples, predict a batch, export it and clear it. With no compatible bundle, inference must be disabled rather than silently use a different feature contract.

**Acceptance Scenarios**:

1. **Given** a compatible serving artifact and its metadata, **when** Page 06 opens, **then** the model identity, actual input count, matching test R² and empirical band come from that artifact rather than static Top-2 claims.
2. **Given** supported role/category relationships and role-specific experience statistics, **when** the category or title changes, **then** title options and experience minimum, maximum and median default update together; no free-text scenario field is exposed and stale incompatible values cannot be queued.
3. **Given** a validated scenario, **when** Add is selected, **then** a stable scenario identity, title, category and experience enter the session queue; a validation badge describes supported inputs without claiming statistical in-distribution certification.
4. **Given** verified benchmark records, **when** a quick-load button is selected, **then** the original record identity and actual salary are preserved and the same approved validation policy is applied; no actual salary becomes a model input.
5. **Given** a nonempty valid queue and compatible artifact, **when** Run is selected, **then** all rows are revalidated before prediction, results are associated with that queue snapshot and model, and labelled salary bars, lower/upper whiskers and optional actual markers precede the results table. Actual markers remain at their true values even outside the band.
6. **Given** successful results, **when** the growth view is opened, **then** predictions across experience values use the same approved model and input policy, disclose the valid domain and do not imply a causal or necessarily monotonic salary trajectory.
7. **Given** successful results, **when** export is selected, **then** the downloaded prediction audit contains the same rows, model, point estimates, bounds and known actuals; a separate original-schema scenario export preserves enterprise column order without inventing unknown attributes or presenting predictions as observed salaries.
8. **Given** results exist, **when** the queue changes, **then** stale results cannot be mistaken for a new run; Clear removes both queue and results, and users' sessions remain isolated.

---

### User Story 4 - Generate trustworthy missing evidence offline (Priority: P1)

As the project maintainer, I want a separate offline evidence command that generates only missing or invalidated model evidence, so the three UI pages can consume compatible artifacts without growing the existing training monolith or training during interaction.

**Why this priority**: Provenance-backed train scores, Top-2 serving and benchmark examples are prerequisites for the full requested experience.

**Independent Test**: In an isolated workspace with existing DEV/test and full-model artifacts, generate a versioned supplemental evidence pack, validate its contracts and reload predictions, then repeat unchanged and prove that valid outputs are reused with zero fit calls. Verify original artifacts and `src/ai_job_market/core.py` remain unchanged.

**Acceptance Scenarios**:

1. **Given** compatible source evidence, **when** the offline command runs, **then** it records source/configuration/dependency hashes, seed, split membership, actual estimator settings and generated-file hashes; missing train/validation evidence is evaluated together on identical folds.
2. **Given** the full model's frozen Random Forest settings and declared two-input policy, **when** a Top-2 bundle is generated, **then** preprocessing fits only on each training partition and final DEV, its own CV/test metrics and empirical band accompany it, and test results never choose or retune it.
3. **Given** previously scored historical test rows, **when** three quick-load examples are exported, **then** selection is deterministic and independent of target/error, examples retain real identities and full/Top-2 predictions, and labels disclose inclusion in historical scoring rather than claim pristine status.
4. **Given** valid existing supplemental evidence with unchanged inputs and producing contract, **when** the command is repeated, **then** it verifies and reuses it without fitting; invalidated components are regenerated or explicitly unavailable, not silently adopted.
5. **Given** generation or compatibility validation fails, **when** the command exits, **then** no incomplete pack becomes active, previous artifacts remain intact, and the UI reports actionable unavailable states without invoking the command itself.

---

### User Story 5 - Scan a visually prioritized analytical dashboard (Priority: P1)

As an executive or auditor, I want color and chart size to consistently communicate meaning and importance so that I can find the primary result first and distinguish training, validation, test, baseline and actual values without reading every legend.

**Why this priority**: The baseline implementation is functionally correct but gives too many charts equal visual weight. The stakeholder intent requires visual-first hierarchy, not merely chart-first ordering.

**Independent Test**: Open Pages 04–06 at 1280×800 and 1440×900 with the real evidence pack. Confirm the first decision chart in each section receives the dominant width, supporting charts are compact or paired, and identical evidence roles use identical colors. Alter a series label/order in a fixture and verify role-based color remains stable.

**Acceptance Scenarios**:

1. **Given** Page 04 comparison evidence, **when** the overall chart renders, **then** validation error is shown as blue/teal columns, training error as an orange line with markers, and the Dummy floor as a dashed red reference. Model status is additionally communicated by text/icon, never color alone.
2. **Given** Page 05 generalization evidence, **when** CV and historical-test errors render, **then** historical-test error is the red column series and DEV-CV is the orange comparison line. R² remains in a separate compact unitless chart rather than being stretched to the same prominence as the primary dollar-error chart.
3. **Given** Page 06 prediction results, **when** the prediction chart renders, **then** prediction bars, uncertainty whiskers and known-actual markers have stable, distinguishable semantic colors and retain direct labels/shape differences.
4. **Given** a section with one primary and one or more supporting charts, **when** it renders on desktop, **then** only the primary P1 decision chart may occupy the full content width. P2 supporting charts use a bounded-width container or balanced two-column row; P3 diagnostic/reference charts use a compact column, expander or on-demand view.
5. **Given** a chart is compacted, **when** labels would clip or overlap, **then** the chart moves to the next larger layout tier or reduces visible categories through an evidence-preserving view; required evidence is not hidden merely to satisfy compactness.
6. **Given** the active theme changes, **when** charts render, **then** semantic series colors remain legible with sufficient contrast, while warnings/errors continue to use native Streamlit status components and no fragile internal CSS selectors are introduced.

---

### User Story 6 - Read the report without pipeline knowledge (Priority: P1)

As a third-party reviewer who did not build the model, I want readable charts, plain-language metric guidance and evidence-derived conclusions so I can identify each page's main finding, supporting numbers and limitations without opening source files or CSVs.

**Why this priority**: Stakeholder review found overlapping labels and numbers that could not be read. Correct evidence is not useful when its presentation obscures values or assumes specialist context.

**Independent Test**: Give a reviewer only the running Pages 04–06 at 1280×800 and 1440×900. Without opening downloads, they must identify each page's purpose, main finding, two supporting values and primary limitation. Browser inspection must show no clipped/overlapping required labels.

**Acceptance Scenarios**:

1. **Given** any Page 04–06 page, **when** it opens, **then** a short page brief states the business question, evidence scope and one-sentence current takeaway before detailed tabs/blocks; a collapsed `How to read these metrics` guide defines the technical terms used on that page.
2. **Given** a major analytical section, **when** its chart is shown, **then** a plain-language question/description appears before it and a derived conclusion appears immediately after it using `Finding`, `Why it matters`, `Limit`, and optional `Decision/use`; conclusions change with artifact evidence and never rely on hardcoded winner/value prose.
3. **Given** direct chart labels, **when** the chart renders at either target desktop width, **then** required values are visible without hover, do not overlap each other, and are not clipped by the plot boundary. Dense supporting values remain available in hover and the following formatted table.
4. **Given** a chart with long model, feature or scenario labels, **when** compact width would make it unreadable, **then** it uses a suitable orientation, wrapped/short display label, larger height/margin or lower direct-label density while preserving full names in hover/table.
5. **Given** monetary, percentage, R², duration and count values, **when** displayed, **then** units and precision are consistent, axes avoid scientific notation, negative R² remains visible, and compact notation does not replace exact evidence in the table/download.
6. **Given** Page 04, **when** a third party reads it, **then** conclusions identify the winner and runner-up gap, improvement versus Dummy, worst/most variable fold and the distinction between candidate ranking and saved-model selection.
7. **Given** Page 05, **when** a third party reads it, **then** conclusions explain CV-to-test changes by metric/direction, historical exposure, q90/coverage and tail risk, full-versus-Top-2 comparison, and tuning limitations without calling lower error an accuracy gain.
8. **Given** Page 06 before and after prediction, **when** a third party reads it, **then** the page explains the two active inputs and strict domain, then summarizes scenario count/range, uncertainty basis, known-actual comparison availability and any extrapolation; it states the safe use is scenario comparison rather than a salary guarantee.
9. **Given** evidence needed for a conclusion is missing, **when** the section renders, **then** it displays `Conclusion unavailable` with the missing evidence and remediation rather than a generic or stale conclusion.
10. **Given** responsive stacking or a tab change, **when** charts rerender, **then** text remains readable and no conclusion refers to a hidden/different model, population or queue snapshot.

### User Story 7 - Audit training and evaluation without reading code (Priority: P1)

As a model reviewer, I want consistent emphasis in Page 04–05 tables and detailed, collapsible explanations of temporal validation and tuning so I can understand what was trained, how it was evaluated, why each procedure was used and which result governed the decision.

**Why this priority**: Tables currently use mixed presentation patterns, while the most consequential training procedures are summarized too briefly for an independent reviewer.

**Independent Test**: Open Pages 04 and 05 with the current evidence pack. Confirm that compact decision tables use one documented bold/italic grammar, all numbers use the same units/precision as adjacent charts, temporal-fold and tuning guides explain the actual implementation, and each guide provides complete evidence downloads. Repeat with missing or incompatible audit logs; the page must preserve validated UI evidence and label the optional trace unavailable rather than blend runs.

**Acceptance Scenarios**:

1. **Given** a Page 04 or Page 05 decision table, **when** it renders, **then** bold identifies the governing winner/applied value, italic identifies reference/context values, and a nearby legend explains those meanings; detailed tables use the same names, units, precision and highlight rule without displaying raw Markdown.
2. **Given** temporal-fold evidence, **when** the Page 04 validation guide is expanded, **then** it explains what is trained, the chronological sliding-window construction, fold-local preprocessing, the no-row-leakage intent, the metrics and the reason for temporal rather than random evaluation. It also discloses that this implementation uses adjacent fixed-size row blocks rather than ordinary randomized or expanding-window K-fold validation.
3. **Given** tuning evidence, **when** the Page 05 tuning guide is expanded, **then** it explains the four-config anchor search, sequential one-parameter sweeps, repeated temporal-fold scoring, descending CV R² ranking, fixed values carried between stages, saved-configuration source and non-nested limitation. It does not present unsupported rationale text as scientific proof.
4. **Given** current supplemental tables, **when** either guide is viewed, **then** all displayed counts, periods, settings and scores derive from the active evidence pack or saved model contract; model-family ranking by MAE remains distinct from tuning ranking by R².
5. **Given** a compatible complete training-audit run under the fixed workspace log directory, **when** audit evidence is requested, **then** its run identity and provenance are visible and the relevant manifest, fold/trial CSVs and complete JSONL trace are downloadable. If compatibility cannot be established, the optional trace is unavailable with a reason and is never merged into active-pack calculations.

---

### Edge Cases

- Missing, corrupt, incompatible or mixed-run metrics, models and metadata: identify the affected artifact and remediation; never train, fabricate values or silently substitute a different model.
- A candidate family or Dummy baseline is absent, rankings change, folds differ, a metric is nonfinite, or train MAE is zero: preserve valid evidence and show unavailable comparisons explicitly; no division by zero.
- Month labels overlap between adjacent chronological row blocks: report actual boundaries and row counts, not strict month separation or a fabricated 200-row final block.
- Empty diagnostic filters, missing tuning stages, tied settings or undefined R²: distinguish unavailable evidence from zero and identify displayed population.
- A title has no valid category mapping or finite experience statistics; minimum equals maximum; a previous session contains an obsolete selection: block invalid submissions with a specific corrective message.
- A benchmark violates the role guard, or a proposed 0–15-year growth curve exceeds observed support: strict validation remains the default. Only explicitly acknowledged, labelled experience-domain exceptions are allowed under FR-017; never silently alter benchmark inputs.
- Unknown actual salary: leave actual/error cells empty and omit actual markers. Zero actual salary: percentage variance unavailable, not infinity. Negative or nonfinite supplied actuals: invalid evidence.
- Empty queue, repeated scenarios, oversized queue or tampered session values: disable Run for empty/invalid queues, retain unique row identities for duplicates and enforce the documented queue limit before inference.
- Predicted lower bound below zero: show a zero-clipped lower bound and disclose clipping; do not force actual markers into the interval.

## Requirements *(mandatory)*

### Functional Requirements

#### Evidence and presentation

- **FR-001**: Scope includes Pages 04–06, their supporting presentation/validation/inference contracts and a separate offline supplemental-evidence workflow. New evidence-generation logic MUST live in new focused `.py` files, never in `src/ai_job_market/core.py`; reuse its existing public helpers without changing that module. Preserve the original pipeline, splits, full-feature model and selection/tuning policy. The declared Top-2 experiment and additional diagnostic evaluations are additive, not a replacement training system.
- **FR-002**: Every displayed metric, count, parameter, period, uncertainty band and model label MUST trace to compatible artifacts. Illustrative numbers in the input document MUST NOT override verified evidence. Provenance must identify source file, run where available, model/configuration, partition and units.
- **FR-003**: Each analytical section MUST place its primary chart before its detailed table, with a compact KPI ribbon permitted first. Bars, comparison-line points, interval endpoints and reference lines MUST have legible direct labels; dense diagnostic scatter points may use hover details to avoid unreadable overlap. Color MUST NOT be the only distinction.
- **FR-004**: Prose MUST be limited to concise interpretation, evidence caveats and collapsed methodology; critical missing-evidence and uncertainty warnings MUST remain visible. Mixed USD and unitless metrics MUST use separate panels or clearly labelled separate axes.

#### Page 04

- **FR-005**: Page 04 MUST provide the five-chip ribbon and four tabs specified in US1, ranking all candidates by recorded mean temporal validation MAE and deriving winning/rank labels rather than hardcoding them.
- **FR-006**: The overall comparison MUST show validation-versus-training MAE, the Dummy reference, validation/train error ratio and validation R² where provenance supports them. The executive table MUST include rank, family, role, validation/train MAE, MAE gap, validation/train R², validation RMSE, MedAE, fit time and an evidence-qualified status.
- **FR-007**: Fold timelines and deep-dives MUST use recorded chronological order, actual sample counts and periods. Random Forest MUST show fold performance and reliance drift; Gradient Boosting MUST show head-to-head fold performance and MedAE comparison. Technical chips MUST distinguish raw features, encoded columns, model settings and measured runtime.
- **FR-008**: Baseline controls MUST contrast Linear, Ridge and Dummy with the nonlinear reference. Fit badges MUST be identified as diagnostic interpretations with explicit criteria and insufficient-evidence states, not scientific proof. Near-singularity, memorization, shallow-tree root causes and absence of bias MUST NOT be asserted from error gaps alone.

#### Page 05

- **FR-009**: Page 05 MUST provide five sections: generalization scoreboard; actual/predicted and residual diagnostics; tuning; feature reliance and feature-set comparison; uncertainty and audit examples, with the six-chip ribbon in US2.
- **FR-010**: The scoreboard MUST compare the final applied configuration's CV MAE, MedAE, RMSE and R² with its recorded test metrics, display absolute and relative deltas, and identify the metric behind each gap. Relative deltas MUST be unavailable when the reference is zero; no change in error may be called an accuracy percentage gain.
- **FR-011**: Tuning MUST expose the four recorded sweep spaces and MAE/R² sensitivities, a compact summary of applied values, and a collapsed explanation of the actual method without textbook search-method comparisons. Missing stages MUST not be invented; sensitivity results MUST not silently redefine the deployed settings.
- **FR-012**: Feature reliance MUST distinguish raw permutation MAE increases (USD, with variability) from encoded importance and temporal drift. Top-2/full-feature metrics MUST refer to comparable populations, and greater input granularity MUST NOT be claimed as proof of production fitness or causal business levers.
- **FR-013**: Trust reporting MUST identify the q90 calibration population, observed coverage denominator and RMSE/MedAE tail ratio where defined. Audit rows MUST retain real source identities, inputs, actuals, model-specific predictions and provenance; pristine status requires documented exclusion history.

#### Page 06

- **FR-014**: Page 06 MUST provide benchmark quick-load, controlled builder, validation queue/action bar and chart-first results. The intended Top-2 mode MUST run only with a verified two-input model and matching evaluation/interval evidence, never a 13-feature bundle passed two columns.
- **FR-015**: Generate missing or unverifiable train-score diagnostics, fixed-configuration Top-2 model/evaluation evidence and three traceable benchmark examples offline in new focused modules. Preserve source artifacts in a separate versioned supplemental namespace, with fingerprints and producer-consumer validation. Reuse compatible existing evidence; regenerate or explicitly invalidate affected evidence when its producing inputs or contract change. Previously exposed test examples MUST be labelled historical benchmark examples, not pristine reserved records; no new data acquisition or test repartition is included.
- **FR-016**: Scenario input MUST use metadata-approved categorical choices and title/category pairs observed in the development partition, labelled observed rather than logically verified. Experience limits and median defaults MUST use finite title-level development values only, never targets or test-derived statistics. Changes MUST immediately invalidate/reset dependent incompatible selections. No free-text input, arbitrary uploaded scenarios, invented fallback categories or curated taxonomy is permitted.
- **FR-017**: Uniform role-specific experience bounds MUST be the default for manual scenarios, quick-load benchmarks and growth curves. Manual entries remain strictly bounded. An opt-in acknowledgement MAY permit an experience-only exception for an immutable identified benchmark or a growth-curve point within the declared 0–15-year exploratory domain. Such records/points MUST carry an extrapolation label, reason and uncalibrated-coverage warning through charts, tables and audit exports; unknown categories, unobserved pairs, nonfinite values, missing policy and absent models remain invalid and cannot be bypassed.
- **FR-018**: All queue records MUST be validated on entry and again immediately before inference, including schema, finite numeric types, enum membership, role/category consistency and approved experience policy. Limit each session to 100 scenarios by default; invalid batches MUST fail before any prediction and report affected rows.
- **FR-019**: Queue mutations MUST invalidate prior results; duplicates MAY be retained with unique scenario identities. Run MUST require a valid nonempty queue and compatible artifact. Clear MUST remove queue and results, without writing training data or artifacts.
- **FR-020**: Results MUST retain scenario identity and use the selected model's actual predictions and empirical q90; lower bound is max(0, prediction − q90), upper bound is prediction + q90. Actual salaries remain audit-only. Signed variance is 100 × (prediction − actual) / actual for positive known actuals, and absolute error is labelled separately. Rounding occurs for display/export, not intermediate arithmetic.
- **FR-021**: Prediction bars, labelled endpoints, actual markers, growth curves, tables and downloads MUST agree on the model and queue snapshot. Growth predictions MUST respect FR-017, identify fixed inputs and disclose that the curve is a model scenario rather than causal growth.
- **FR-022**: Users MUST be able to download a prediction audit and a separate original-enterprise-schema scenario file. Unknown original attributes MUST remain empty; predictions and empirical intervals MUST NOT be misrepresented as original observed data. The exact ordered export columns and precision MUST be locked in the plan's consumer contract before implementation.
- **FR-023**: Missing/incompatible evidence MUST produce actionable, local error states, preserving unrelated valid sections. No UI action may fit, tune or regenerate an artifact. Top-2 metrics, benchmark actuals or successful serialization checks MUST never use fabricated defaults.
- **FR-024**: Documentation MUST describe chart sources, model/partition identity, validation/domain policy, queue lifecycle, export semantics, unavailable states and scientific limitations. Automated validation MUST precede implementation of changed behavior.
- **FR-025**: Supplemental evaluation MUST freeze model configurations and feature policies before reading test targets, record temporal fold identities, and compute train/validation scores from the same fitted fold models. It MUST keep candidate-family evidence distinct from final full/Top-2 evidence and MUST NOT change model selection based on historical test results. Report all five frozen candidates' CV/test metrics as diagnostics, not a fresh selection experiment.
- **FR-026**: The offline workflow MUST publish only validated, complete components with a run manifest, explicit dependency identity and availability state. Failures MUST not overwrite baseline evidence or advertise unfinished models. Matching unchanged runs MUST be reused without retraining; ordinary UI actions MUST never call the producer.
- **FR-027**: Charts MUST use a shared role-based semantic palette rather than Plotly defaults or model-order-dependent colors. Validation/prediction primary values use blue/teal; training/DEV comparison values use orange; historical-test error columns and failure/threshold references use red; known actual observations use a distinct red marker with a different shape; neutral/reference evidence uses gray. The same evidence role MUST retain the same color across Pages 04–06. Every meaning MUST also be communicated by legend text, labels, line style or marker shape so color is never the sole encoding.
- **FR-028**: Every chart MUST declare a visual priority. P1 decision charts render first and may use the full available content width. P2 supporting charts MUST use a bounded-width container or a balanced two-column layout on desktop. P3 diagnostics/reference charts MUST be compact, collapsed or explicitly requested. A chart MUST NOT use full width merely because `width="stretch"` is available; stretching is limited to its assigned container. Layout may promote a chart when compact sizing would make direct labels unreadable, but MUST record that exception in the view contract and preserve mobile stacking.
- **FR-029**: Each page MUST begin with a plain-language report brief containing purpose/question, evidence scope, current evidence-derived takeaway and key limitation. Each page MUST provide a collapsed metric guide defining only the terms it uses, including MAE, MedAE, RMSE, R², CV, residual and q90 where applicable.
- **FR-030**: Each major section MUST state the question answered before its primary visual and show a deterministic conclusion immediately afterward. Conclusion data MUST be derived from validated tables/snapshots and structured as `Finding`, `Why it matters`, `Limit`, and optional `Decision/use`; missing dependencies yield a named unavailable conclusion.
- **FR-031**: Figure builders MUST own chart-specific width tier, height, margins, axis automargin, tick format, label policy and dense-data fallback. Shared rendering MUST preserve figure-specific geometry and MUST NOT force one fixed margin profile over every chart.
- **FR-032**: Required direct labels MUST remain visible and non-overlapping at 1280×800 and 1440×900. Primary bars retain direct labels. Dense comparison lines label only evidence-selected key points when all-point labels would collide; full values remain in hover and the immediately following table. Scatter plots with more than 20 points MUST NOT render per-point text labels.
- **FR-033**: Long categorical labels MUST use horizontal orientation, wrapping, deterministic display abbreviations or larger reserved margins as appropriate. Full original labels MUST remain available in legends, hover or tables. Importance views MUST use an explicitly stated Top-N display while downloads retain all rows.
- **FR-034**: Numeric presentation MUST use consistent units and readable precision: whole USD or a disclosed compact `$k` axis, R² to three decimals, percentages to one decimal, durations with an appropriate unit and counts as integers. Scientific notation MUST be disabled for ordinary salary/error axes. Negative R² and zero/reference lines MUST not be cropped.
- **FR-035**: Page 04 conclusions MUST include winner/runner-up separation, Dummy improvement and temporal weakness; Page 05 conclusions MUST include final-config CV/test deltas, q90/coverage/tail risk, matched full/Top-2 comparison and tuning/exposure limits; Page 06 conclusions MUST include active input/domain scope and, after Run, batch range, uncertainty basis, known-actual availability and exception count. Conclusions MUST avoid unsupported causes, universal fit verdicts or compensation advice.
- **FR-036**: Detailed evidence MUST follow progressive disclosure: page brief and primary answer first, supporting diagnostics next, formatted table/download after the related chart, and methodology/raw detail collapsed where it is not necessary for the main conclusion. No required warning may be hidden in an expander.
- **FR-037**: Page 04 and Page 05 tables MUST follow one table presentation grammar. Small decision tables use Markdown-capable static tables: bold marks the governing winner, selected configuration or decision metric; italic marks baselines, references or contextual scope. A visible legend defines both. Large/detail tables remain interactive, use reader-facing headers and consistent formatting, and MUST NOT expose literal Markdown syntax as cell text.
- **FR-038**: The same metric MUST use the same displayed precision in KPIs, charts, compact tables and detailed tables on a page: whole USD, R² to three decimals, percentages to one decimal, durations to three decimals unless a longer unit is named, and counts as integers. Winner/highlight rules MUST use unrounded source values and deterministic tie handling.
- **FR-039**: Page 04 MUST place a high-priority collapsed temporal-validation guide before analytical tabs. It MUST explain what candidate pipelines and features are trained, how chronological row blocks form train/next-block validation pairs, that preprocessing is fit within each train fold, what metrics are aggregated, why temporal validation is used, and how this differs from random/expanding K-fold. Actual fold count, row counts and periods MUST derive from active supplemental evidence.
- **FR-040**: Page 05 MUST place a high-priority collapsed tuning guide within the tuning reading path before sensitivity charts. It MUST explain the actual four-anchor plus sequential coordinate-sweep procedure, trial counts/search spaces, fixed/carried values, five-fold temporal scoring where evidenced, descending CV R² selection rule, relationship to the saved configuration, no locked-test selection, and inherited non-nested limitation. Unsupported legacy `tuning_rationale` prose MUST remain excluded.
- **FR-041**: Page 04 candidate ranking and Page 05 tuning selection MUST remain visibly distinct: candidate families rank by mean validation MAE (lower is better), while inherited tuning trials rank by CV R² (higher is better). Bold/italic styling MUST never imply that a supporting sweep winner automatically replaced the saved model.
- **FR-042**: Current active-pack CSVs remain authoritative for Page 04–05 scientific displays. A training-audit trace MAY be exposed only from the fixed workspace log directory after manifest completeness, source fingerprint, safe path and checksum validation. Its run identity and status MUST be shown; audit values MUST NOT silently replace or combine with active-pack values.
- **FR-043**: The temporal-validation and tuning guides MUST provide related evidence exports. At minimum this includes current fold metrics and membership for Page 04; tuning summary and every available stage table for Page 05; and, when FR-042 compatibility passes, the audit manifest, relevant exported CSVs and complete JSONL trace. Missing optional logs MUST not disable the guides or trigger training.

### Constitutional Requirements *(mandatory)*

- **Data boundary**: Target remains `annual_salary_usd`. Preserve existing raw-feature and temporal split policies; no target or actual salary enters scenario features. Current metadata reports 1,201 development rows and 298 March 2026 test rows; these are observed evidence, not permanent constants. Do not repartition already exposed test records and call them newly pristine. Role guards use target-independent statistics from an explicitly declared population; their policy is not a new training transformation.
- **Artifact contract**: Existing source packs remain `outputs/04_model_comparison/`, `outputs/05_best_model/`, `outputs/06_salary_prediction/` and `artifacts/metadata.json`, `artifacts/feature_contract.json`, `artifacts/model_bundle.joblib`, with DEV/test inputs from `outputs/02_data_ready_for_ml/`. The approved scope adds a versioned supplemental contract rather than modifying those files. Update its new producer, consumers, tests and documentation together; regenerate missing/invalidated supplemental artifacts before reliance. Preserve old outputs and reject unsupported optional fields rather than silently treating them as reproducible. No dependency, main-pipeline configuration or source-data changes are planned; never retrain solely for a UI demo.
- **Streamlit boundary**: Pages remain presentation/orchestration consumers. Validation, inference and display calculations must be independently testable; fitting/tuning stays offline. Reading an artifact is not permission to claim its unsupported fields are reproducible.
- **User input validation**: Validate widget choices, filters, slider values, queue rows, benchmark references and any model/run selectors before use; bounded enum selection only, no arbitrary paths or uploads. Preserve current authorization boundaries, never evaluate user-supplied model paths, and isolate session queues. Existing credentials/authentication behavior is outside this feature's scope.
- **Scientific interpretation**: Distinguish model-family CV, tuned CV, historical test metrics and exploratory diagnostics. Empirical bands are not guaranteed future coverage; exception cases have no validated extrapolation coverage. Feature importance is reliance, not causation or fairness; academic/synthetic-looking data does not establish production fitness. Supplemental candidate test diagnostics do not retroactively repair inherited test exposure, non-nested tuning or selection-policy discrepancies. Top-2 settings are fixed, with no new tuning or test-driven promotion.
- **Documentation impact**: Add a page usage/evidence guide and update relevant application guidance. Keep the stakeholder source document unchanged as input; record corrections and scope decisions in this spec and its evidence review. Reference existing generated charts where available; interactive views derive only from verified output evidence.
- **Verification evidence**: Require failing-first tests for evidence mappings, invalid inputs, missing artifacts, stale session state, exports and interval calculations; producer-consumer integration checks must prove model/feature compatibility without fitting. Exercise affected application pages with existing approved output packs and missing/corrupt fixtures. Browser review must verify chart-first ordering, labels and keyboard-accessible controls. No application tests or training have been run during this specification phase.
- **Graphify/Karpathy review**: Existing graph queried to locate serving, metadata and evaluation relationships; truncated output was followed by direct source/artifact inspection. Make surgical page/support changes, preserve existing training, and avoid new infrastructure. Documentation-only specification work does not require a graph rebuild; implementation must refresh it or record a valid exception.

### Key Entities *(include if feature involves data)*

- **Evidence pack**: A compatible collection of source artifacts with model/configuration identity, run identity where present, provenance status, feature policy and partition scope.
- **Candidate/fold result**: Model family, chronological fold, sample counts, train/validation metrics, runtimes and evidence-qualified diagnostic interpretation.
- **Tuning trial**: Parameter configuration, evaluation scope, CV metrics and relationship to the final applied configuration.
- **Diagnostic prediction**: Real record identity, actual salary, model prediction and residual/error, linked to its test population.
- **Role constraint**: Allowed category/title pair and finite experience minimum, maximum and median, with reference-population provenance.
- **Benchmark record**: Real source identity and immutable profile/actual, with verifiable inclusion/exclusion history and model-specific prediction evidence.
- **Scenario queue and result batch**: Session-local uniquely identified scenarios, validation state, immutable prediction snapshot, artifact identity, point/bounds, optional actual/error and export representation.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Across Pages 04–06, 100% of displayed scientific numbers and model/partition labels reconcile to approved evidence or a documented display derivation at the displayed precision; missing evidence never yields a fabricated value.
- **SC-002**: Page 04 contains four functional tabs, Page 05 five functional sections and Page 06 four functional blocks; every analytical section with available evidence presents its primary chart before its detailed table.
- **SC-003**: A reviewer can identify the winning family, its mean validation MAE and baseline comparison within 60 seconds, then identify final-model test MAE, test population and uncertainty limitation within another 60 seconds in a scripted acceptance walkthrough.
- **SC-004**: All invalid category/title/experience and tampered-queue cases in the agreed validation matrix are rejected before inference; no scenario-builder control accepts free-text values. Missing serving prerequisites visibly disable the affected action.
- **SC-005**: For batches of 1, 3 and 100 valid scenarios, predictions, interval endpoints, actuals and signed errors agree across charts, table and audit export; Clear leaves no queue or result rows and session-isolation tests show no cross-user data.
- **SC-006**: At desktop widths of 1280 and 1440 pixels, labels for candidate bars, fold points and prediction bounds are readable without hover or clipping; larger batches may use a disclosed paged/scrollable view without dropping table/export records.
- **SC-007**: For three scenarios on a documented reference machine with artifacts already loaded, target completed visible results within one second; record cold-load and warm prediction/render timings separately. The source's sub-0.01-second claim remains an unverified inference aspiration, not an established end-to-end guarantee.
- **SC-008**: All affected page acceptance tests and artifact-consumer integration checks pass without any UI fitting/tuning or changes to preserved training artifacts; missing optional evidence is visibly distinguishable from measured zero.
- **SC-009**: A successful offline generation produces traceable train/validation scores, a reloadable two-input bundle, matching full/Top-2 historical test evidence and up to three real benchmark examples; an unchanged repeat performs zero fits. Every generated artifact has an integrity/provenance entry and the original pipeline module/artifacts retain their pre-change hashes.
- **SC-010**: All default scenarios, benchmarks and curve points obey the same development-derived domain policy. Every allowed experience exception is explicitly acknowledged and labelled wherever shown or exported; no exception bypasses categorical or artifact validation.
- **SC-011**: In automated figure tests, 100% of required training, validation, historical-test, prediction, uncertainty, actual and baseline traces use their contracted semantic color plus a non-color distinction; trace order changes do not change role colors.
- **SC-012**: At 1280×800 and 1440×900, each analytical section has at most one full-content-width P1 chart before its table. P2/P3 charts occupy bounded or paired containers unless a documented label-readability exception promotes them; no direct label is clipped in the accepted layout.
- **SC-013**: Browser bounding-box inspection at both target widths finds zero intersections among required direct labels and zero required labels outside the plot/card viewport across all Page 04–06 tabs and a 10-scenario Page 06 result chart.
- **SC-014**: Automated tests reconcile every conclusion value/direction to its source evidence, prove conclusions update when ranking/deltas change, and produce a named unavailable state when required evidence is absent.
- **SC-015**: In a fresh third-party walkthrough, the reviewer identifies each page's main finding, two supporting numbers and principal limitation within 90 seconds per page without opening a CSV or source file.
- **SC-016**: All displayed tables on Pages 04–06 use reader-facing column names and unit formatting; identity columns remain visible in wide result/audit views, while complete machine-readable data remains downloadable.
- **SC-017**: Every Page 04–05 decision table displays the documented `bold = governing result` and `italic = reference/context` semantics, and automated tests verify the highlighted row/value changes when source ranking or applied settings change.
- **SC-018**: A reviewer can expand the Page 04 guide and correctly identify the train/validation construction, fold-local preprocessing, ranking metric and temporal rationale; they can expand the Page 05 guide and identify the anchor/sweep sequence, trial scoring, selection metric, saved-configuration relationship and non-nested limitation without opening source code.
- **SC-019**: All values repeated across Page 04–05 KPIs, charts and tables agree at the declared precision; winner and tuning-stage highlight decisions are computed from unrounded evidence with deterministic ties.
- **SC-020**: For a compatible complete audit run, 100% of offered log/CSV downloads pass path and checksum validation and display the audit run identity. For missing, incomplete or incompatible logs, no audit-derived value enters a scientific display and no training call occurs.

## Assumptions

- The stakeholder document defines desired information architecture, not permission to hardcode its claims. Evidence corrections required by project scientific-integrity rules are documented in [evidence-review.md](evidence-review.md).
- Q1–Q3 were resolved by the user on 2026-09-19: expand offline generation in new `.py` modules (not the god module); uniform bounds are the common/default path with labelled exceptions allowed; use observed dataset pairs. This authorizes the updated plan/tasks, not their execution. No fabricated pristine records, full-model two-input fallback or validation bypass is authorized.
- Current metadata's option lists, rather than prose examples, define canonical spellings. For example, `Research` is a listed category and `Principal Data Architect` is not a listed title.
- Dynamic title associations and title-level bounds use the persisted DEV partition, intersected with compatible metadata enums. Some medians may be fractional; the plan must preserve them rather than silently round to an integer. Benchmark selection prefers strict-policy eligible historical test rows and may return fewer than three rather than fabricate examples.
- Source-derived metrics are not changed; unit conversion, unrounded display deltas, signed variance and interval formatting are presentation calculations. Any incompatible change to an existing exported meaning requires an explicit consumer-contract migration.
- The proposed 100-row queue cap, separate audit/source-schema downloads and one-second warm user-experience target are reviewable defaults, not measured achievements.
- Existing navigation, authorization and unrelated pages remain unchanged. No new dependencies, custom component system, global redesign, primary-training changes or data collection are assumed. The existing global upload/process control is outside scope and must not be invoked by the new UI workflow or presented as proof that all historical UI paths are training-free.
- “Match the stakeholder file” means match its visual information architecture, semantic series roles and chart-first intent. Unsupported literals and claims in that input—such as pristine examples, a 295-row denominator or causal/proven fit diagnoses—remain corrected by verified evidence and MUST NOT be restored for visual fidelity.
