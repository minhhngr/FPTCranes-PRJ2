# Feature Specification: Readable Training Logs and Evaluation Evidence

**Feature Branch**: `004-readable-training-logs`  
**Created**: 2026-09-18  
**Status**: Draft — specification validated; implementation not authorized  
**Input**: User requests readable, step-by-step training output with four information levels: default levels 1–2 and additional levels 3–4 with `--debuglog`; explain data sources, features, stage counts, train/validation/test splits, folds, model comparison, fit diagnostics and hyperparameter tuning. Keep structured JSON in files rather than the terminal; export large tables to CSV. Add observation/export only, without changing training logic.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Follow Training Without Reading Raw Logs (Priority: P1)

As a person running training, I want readable descriptions and numerical summaries so I can understand which model is running, what it is doing and which data it uses without interpreting raw machine records.

**Why this priority**: The current complaint is terminal readability; the ordinary training experience must be useful without debug mode.

**Independent Test**: Capture a normal training run and verify that each executed training stage has a readable description and numerical summary, with no raw structured event dumps.

**Acceptance Scenarios**:

1. **Given** training starts without `--debuglog`, **When** stages execute, **Then** level 1 describes the stage, model, method and purpose, and level 2 is indented two spaces and reports the available training/validation numbers with units and partition labels; levels 3–4 are absent.
2. **Given** a data preparation or feature-selection stage completes, **When** its summary appears, **Then** the user sees the source identity, input/output row counts, feature count, exclusions and their reasons where known, and a readable description of how the data is used.
3. **Given** a run completes, fails or is interrupted, **When** its final message appears, **Then** the terminal clearly identifies its actual status, last stage and evidence location, without dumping JSON or falsely claiming success.

---

### User Story 2 - Inspect Detailed Data and Fold Evidence (Priority: P1)

As a person investigating training, I want `--debuglog` to add detailed, readable evidence explaining the numbers, partitions and operations behind the default summaries.

**Why this priority**: The user explicitly needs to trace data usage and split counts rather than only see progress messages.

**Independent Test**: Compare normal and debug output for the same fixed input and verify additional fold and operation details without any change in training decisions or results.

**Acceptance Scenarios**:

1. **Given** `--debuglog` is enabled, **When** a training stage runs, **Then** levels 1–2 remain visible, level 3 adds split/fold and trial breakdowns, and level 4 adds detailed evidence and derivations; each child level adds two spaces of indentation.
2. **Given** a fold is evaluated, **When** its details are displayed, **Then** the user can identify its parent stage and model, training/validation counts, available time boundaries, actual feature set, preprocessing fit scope and the source of the reported measurements.
3. **Given** a complete feature list or membership table exceeds the display limit, **When** the debug output summarizes it, **Then** the terminal gives its total count, explains the truncation and links the complete export for that run.

---

### User Story 3 - Understand Model Comparison and Tuning (Priority: P1)

As a person comparing models, I want each candidate's measured performance, fit assessment and tuning history explained with supporting evidence, so I can understand the reported winner without guessing or confusing validation with test results.

**Why this priority**: A readable log that hides evaluation evidence would not meet the user's main scientific requirement.

**Independent Test**: Present known candidate and trial results, including missing train scores and failed trials, and verify the displayed values, evidence references and qualified interpretations.

**Acceptance Scenarios**:

1. **Given** model comparison results exist, **When** the summary is displayed, **Then** every evaluated candidate has a readable entry with available MAE, RMSE, R² and MedAE, partition labels, fold aggregates and variability where available; missing values have an explicit reason rather than a fabricated number.
2. **Given** fit diagnostics exist, **When** a candidate is reported, **Then** the entry shows `overfitting`, `good fit` or `underfitting` only with the diagnostic rule and supporting train/validation measurements; otherwise it shows `insufficient evidence` with the missing evidence identified.
3. **Given** hyperparameter tuning executes, **When** a trial finishes, **Then** its parameters, fold identity, seed and measured validation outcomes are retained; default output reports tuning progress and stage winners, while debug output exposes trial/fold detail or a linked complete export.
4. **Given** tuning ends, **When** final selection is reported, **Then** the user sees the actual selection metric and direction, selected settings, settings actually applied to the final model, and any difference between them without silently correcting the experiment.
5. **Given** the locked test is evaluated only for a selected model, **When** the comparison is displayed, **Then** other candidates' test entries say `not evaluated`, and no extra test scoring is performed to fill the table.

---

### User Story 4 - Retrieve Complete Evidence Without Terminal Flooding (Priority: P2)

As a person reviewing a run, I want complete machine-readable evidence and large tables saved separately, so I can inspect exact records after training while keeping the terminal readable.

**Why this priority**: Complete evidence and readable output must coexist; terminal filtering must not discard the audit trail.

**Independent Test**: Use a result table larger than the display limit and verify that the terminal points to a readable complete export with matching counts, values and run identity.

**Acceptance Scenarios**:

1. **Given** either normal or debug mode, **When** events are recorded, **Then** complete available evidence is retained under the run's log area independently of terminal verbosity; raw JSON is never printed to the terminal.
2. **Given** a table has more than 20 data rows, **When** it is presented, **Then** at most 20 rows appear per table, its total size and omitted count are stated, and a complete CSV is available; a reader can open it without rerunning training.
3. **Given** evidence cannot be saved or an existing evidence file cannot be read, **When** that problem occurs, **Then** a readable warning identifies the affected evidence and corrective action, and the run is marked as having incomplete evidence rather than claiming a complete export.

### Edge Cases

- A stage, model or tuning trial is skipped, fails, or is interrupted: distinguish these states from completion and retain the last available evidence.
- Empty partitions, undefined scores, missing training metrics or absent fit rules: display explicit unavailability and its reason; never substitute zero or invent a fit label.
- Train and validation periods share a month or membership overlaps: report observed boundaries and distinguish date overlap from row overlap; do not claim leakage safety based on labels alone.
- Duplicate row identifiers or unequal fold sizes: preserve actual membership identity and counts, without reordering or resplitting for presentation.
- Very wide feature lists or very large fold/trial tables: wrap readable text and export full detail, with no silent loss of records.
- Redirected output, narrow terminals or terminals without color: preserve readable hierarchy and status without relying on color or interactive rendering.
- Multiple runs or repeated model names: identify run, stage, candidate, trial and fold sufficiently to prevent evidence being confused or overwritten.
- Unwritable output destinations, missing/corrupt exports or invalid command options: explain the error; never silently select evidence from another run.
- Sensitive row content or credentials appear in inputs: exclude them from console and exports; full detail means safe training evidence, not unrestricted data disclosure.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The training command MUST support `--debuglog`. Without it, training information levels 1–2 MUST be visible; with it, levels 1–4 MUST be visible. Warnings and errors MUST remain visible in both modes.
- **FR-002**: Level 1 MUST describe the current stage, model where applicable, method and action. Level 2 MUST report stage counts and available training/validation outcomes. Level 3 MUST explain individual splits, folds and tuning trials. Level 4 MUST provide detailed feature, parameter, membership and numerical-derivation evidence, directly or through linked complete exports.
- **FR-003**: The terminal MUST use a consistent two-space indentation increment per level, descriptive headings, labeled numbers, units and clear start/completion/failure states. Machine event payloads, including raw JSON, MUST NOT be printed in either verbosity mode.
- **FR-004**: Every executed data/training stage MUST report its source identity, input/output row counts and feature counts where applicable. Counts MUST identify their basis, including filtering, exclusions, partition assignment or aggregation. Unknown reasons MUST be labeled unknown.
- **FR-005**: Feature evidence MUST identify the target, actual input features used by each model or feature variant, available transformed feature counts and the partition on which preprocessing was fitted. Summarized lists MUST link to complete evidence.
- **FR-006**: Split evidence MUST report the actual development/test policy, available time boundaries, per-fold train/validation counts and membership references. It MUST describe observed behavior rather than an assumed ideal split and MUST NOT open the locked test earlier for logging.
- **FR-007**: Every evaluated candidate MUST have a comparison entry showing available MAE, RMSE, R² and MedAE, train/validation/test labels, aggregation definition and available fold variability. Unmeasured values MUST be labeled `not evaluated` or `unavailable` with a reason. Salary errors MUST state their currency units and R² MUST be identified as unitless.
- **FR-008**: Every candidate comparison MUST include a fit-assessment field. `overfitting`, `good fit` and `underfitting` MUST be qualified diagnostic indications supported by comparable train/validation evidence and an explicit rule, including thresholds and reference baseline where used. Missing evidence or an undefined rule MUST yield `insufficient evidence`. A high R² alone MUST NOT justify a `good fit` label.
- **FR-009**: Tuning evidence MUST include each executed trial's parameters, seed, fold references, available metrics and outcome, stage winners, actual selection criterion and direction, and the final applied settings. Failed/skipped trials and discrepancies between a reported winner and applied settings MUST be explicit.
- **FR-010**: Summaries MUST distinguish development validation from locked-test evaluation, baseline from tuned results and candidate selection from final evaluation. Logging MUST NOT add training, prediction, splitting or held-out evaluation calls or change feature, parameter or winner selection.
- **FR-011**: Complete available evidence for all four levels MUST be retained in run-specific files in the selected workspace's log area in both modes. Terminal filtering MUST NOT filter out persisted evidence. Each displayed measurement MUST be traceable to its run, stage, model/fold/trial where relevant and evidence record or file.
- **FR-012**: Tables exceeding 20 data rows MUST be exported completely to CSV, with at most 20 displayed data rows per table, the total and omitted counts, and the export path. Stored numerical precision MUST be retained even when terminal values are rounded. Existing complete matching exports MAY be referenced rather than duplicated.
- **FR-013**: Saved evidence MUST include run identity, source path or identifier and content fingerprint, relevant configuration/seed, stage order and final status. Separate runs MUST NOT overwrite each other's evidence. Existing model bundles and downstream artifact contracts MUST remain compatible.
- **FR-014**: Evidence write/read failures MUST produce actionable, sanitized warnings and mark evidence completeness accurately. Logging failure MUST NOT change model decisions, mask a training exception or report a failed/interrupted run as successful.
- **FR-015**: User-controlled debug options, output destinations and any evidence file references used for reading MUST be validated before use. Invalid command options MUST fail clearly before training; unsafe destinations MUST be rejected without writing outside the authorized workspace. No new unrestricted file-reading interface is required.
- **FR-016**: Console and persisted evidence MUST exclude credentials and unnecessary sensitive/raw row values. Membership references and safe source identifiers MUST be sufficient to trace evidence without duplicating the raw dataset.
- **FR-017**: The existing training behavior MUST remain unchanged except for additive observation, exports and terminal presentation. No refactoring of model logic, new search, automatic training fixes, dependency replacement mandate or UI training path is in scope.
- **FR-018**: User documentation MUST explain the four levels, `--debuglog`, evidence locations, CSV reading, metric units, fit-assessment limitations, failure handling and the affected training stages. The planning phase MUST inventory affected functions and artifact producers/consumers before any implementation.

### Constitutional Requirements *(mandatory)*

- **Data boundary**: Observe the existing salary target `annual_salary_usd`, actual feature variants, development/locked-test boundaries and training-only preprocessing scopes. Preserve data ordering, seeds, fit/predict/split calls and selection behavior. Report suspected leakage or boundary discrepancies without introducing additional test exposure or silently repairing them.
- **Artifact contract**: Add presentation and run-scoped evidence only. Existing trained bundles, metric tables and reporting consumers remain compatible. New evidence must retain provenance, units, partition identity and explicit links to existing source artifacts when reused.
- **Streamlit boundary**: No UI changes or training in the presentation application. This feature concerns the offline training terminal and its evidence files only.
- **User input validation**: Validate the debug flag, authorized workspace/output paths and any consumed export's existence, format, required fields and run identity. Reject unsafe paths; expose corrupt or mismatched evidence as unavailable, not as a replacement run. Uploads, credentials and prediction inputs gain no new interface in this feature.
- **Scientific interpretation**: Fit assessments are evidence-backed heuristics, not proof of causality or production readiness. Never infer underfitting/overfitting solely from a single held-out score. The prior plan documents inherited candidate-test coverage and tuning-selection policy gaps; these MUST be reconciled with constitution VI.2/VI.4 during planning before implementation approval. This spec neither waives those requirements nor authorizes changing training to satisfy them.
- **Documentation impact**: Update the training operating/audit guide with terminal examples, level semantics, stage/count explanations, comparison/tuning interpretation, safe export inspection and links to existing diagnostic charts. Do not regenerate model results or charts merely to write the spec.
- **Verification evidence**: Before implementation, define failing-first acceptance tests for verbosity, evidence consistency, large exports, missing diagnostics and failures. Verification must exercise normal/debug offline entrypoints with fixed inputs in an isolated workspace and establish unchanged call counts, selected settings and scientific results, without overwriting release evidence.
- **Graphify/Karpathy review**: Before non-trivial implementation, inspect affected source relationships and verify them against current code; choose the smallest observation-only change. This specification-only phase uses the existing plan as context and makes no claim that a new code graph review or runtime validation has been performed.

### Key Entities *(include if feature involves data)*

- **Training Run**: A uniquely identified execution with source/configuration provenance, verbosity choice, ordered stages, completion status and evidence-completeness status.
- **Stage Observation**: An action and its purpose, hierarchy level, input/output counts, units, status and supporting evidence references.
- **Data Partition / Fold**: An actual training, validation or locked-test subset with row count, temporal bounds and safe membership identity.
- **Model Evaluation**: Candidate identity, features, parameters, partition-specific metrics, fold aggregates and qualified fit assessment with its evidence/rule.
- **Tuning Trial**: A tested parameter choice linked to its stage, model, folds, seed, measured outcomes and relationship to final applied settings.
- **Evidence Export**: A run-scoped structured record or complete table with path, record count, precision, provenance and completeness status.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: In acceptance runs, 100% of executed training stages have a readable action/status and applicable numerical summary; normal mode displays no level 3–4 details, and debug mode exposes all four levels directly or through explicit complete-evidence references.
- **SC-002**: Neither normal nor debug terminal captures contain raw structured event dumps. Every table over 20 rows has a complete accessible export, correct total/omitted counts and no more than 20 displayed data rows.
- **SC-003**: Every displayed count and score in the acceptance fixtures reconciles with its saved source evidence, allowing only declared display rounding; every evaluated model has a comparison entry and either a supported fit indication or an explicit insufficient-evidence explanation.
- **SC-004**: A reviewer can identify the model, source data, feature set, partition counts, selection basis and actual final parameters within five minutes using the terminal transcript and its linked evidence, without inspecting training source code.
- **SC-005**: On fixed-input verification runs, normal and debug modes preserve stage order, split membership, model-call counts, selected features/settings and scientific outputs against the observation-free baseline; score comparisons use a declared numerical tolerance rather than visual judgment.
- **SC-006**: All injected export-failure, missing-metric, failed-trial and interruption scenarios communicate the correct training/evidence status without fabricated values or false success.

## Assumptions

- `--debuglog` adds levels 3–4 rather than replacing the ordinary levels. The four levels are presentation depth, distinct from warning/error severity.
- “Cách 2 ô” means indentation by two spaces, not a two-column layout. Each deeper level adds another two spaces.
- English training output remains consistent with the earlier training-audit feature; this request changes readability and completeness, not localization.
- “Show everything” means all safe, available training evidence. It does not mean unbounded terminal dumps, raw confidential rows, per-tree internals or additional model evaluations. Large details may be inspected through linked exports.
- A 20-data-row terminal limit is the initial usability default. It bounds each table, not the number of executed stages or model summaries.
- Existing metrics, events and artifacts are the evidence sources. Descriptive arithmetic on existing evidence is allowed; extra fit/predict/split calls solely to obtain missing metrics are not. If paired training/validation evidence or a defensible rule is absent, the fit assessment remains `insufficient evidence`.
- Logging/presentation/export additions are permitted; altering underlying training algorithms, preprocessing, feature policy, splits, search space or selection decisions is prohibited by this request.
- The earlier feature at `specs/001-training-step-logs/` is context, not proof of the current working tree's completeness. This feature supersedes its terminal JSON mirroring behavior only where necessary, retaining complete file evidence. Neither its files nor in-progress code are modified in this phase.
- Choice of logging/rendering library, including whether to use Rich, belongs to research/planning. No new library is required by this specification.
- No training run, application test or code change is authorized by creation of this spec. Research/planning must resolve evidence availability and inherited governance findings; implementation requires explicit approval of tasks.
- Follow-up instruction (2026-09-18): prepare a source-accurate plan, automatically perform a post-plan research/coverage review, then generate tasks without intermediate approval. This authorizes preparation of those artifacts, not execution of implementation tasks.
- This is an educational project: do not add a separate security/hardening workstream. Keep existing validation and minimal safe file handling; do not remove repository-mandated safeguards.
- Complete step logging covers every executed workflow stage and explicit salary-training operation. Branch A includes representation/evaluation/selection substage boundaries and complete available result tables; estimator-internal iterations and individual resample calls are not separately traced. The implementation coverage inventory must make this boundary explicit.
