# Feature Specification: English Training Step Logs and Audit Evidence

**Feature Branch**: `003-training-step-logs`  
**Created**: 2026-09-18  
**Status**: Draft — specification only; implementation not approved  
**Input**: Inspect the current training and fold behavior without correcting or redesigning it. Add detailed step-by-step logs to both the terminal and per-run `.logs` files. All added code and log messages must use English. Preserve outputs and document evidence for follow-up. R² = 0.85 is a reference only, not a performance requirement.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Follow training live and replay it later (Priority: P1)

As the person running the existing offline training workflow, I want to see what each training step is doing and retain the same evidence after the terminal closes.

**Why this priority**: Stage-level progress alone does not explain which model, trial, fold, or operation is currently running.

**Independent Test**: Run a bounded training fixture and compare terminal events with the saved run log, checking successful and failed operations.

**Acceptance Scenarios**:

1. **Given** an existing training run, **When** it starts, **Then** the terminal and a unique `.logs` file identify the run, input data identity, configuration, seed, and log location.
2. **Given** a model, trial, and fold, **When** preprocessing/fit, prediction, and scoring execute, **Then** start/completion events identify the operation, context, counts, elapsed time, and available metrics in English. If preprocessing and estimator fitting are one existing operation, the log labels it as a combined operation rather than inventing separate timings.
3. **Given** an existing tuning sequence, **When** each candidate completes, **Then** its actual parameters, fold-level results, aggregate results, and the actual selection decision are recorded without changing the sequence or decision.
4. **Given** a run that fails or is cancelled, **When** execution ends, **Then** completed events remain readable, the last active step and failure/cancellation are recorded, and no success is claimed.

---

### User Story 2 - Verify how folds are actually used (Priority: P2)

As a reviewer, I want inspectable fold evidence across comparison, tuning, and other existing salary-training evaluations so I can check consistency without changing the experiment.

**Why this priority**: Equal fold counts alone do not prove equal training and validation membership.

**Independent Test**: Compare logged membership and counts with the exact partitions consumed by the existing training calls on a deterministic fixture.

**Acceptance Scenarios**:

1. **Given** a temporal split, **When** it is used, **Then** evidence records its ordering rule, requested and effective fold counts, block size, actual row membership, train/validation counts, and minimum/maximum periods.
2. **Given** repeated use of a split, **When** models or trials consume it, **Then** each use references an identifiable split and fold so identical or differing memberships can be verified.
3. **Given** same-month boundaries, remainder rows, or unexpected overlaps, **When** the audit encounters them, **Then** it reports the observed facts and distinguishes row overlap within a fold from period overlap and legitimate reuse across folds; it does not repair the split.
4. **Given** a selected model, **When** the existing final evaluation executes, **Then** full-development fitting and locked-test evaluation are labeled separately from temporal CV; logging does not trigger an additional fit or test evaluation.

---

### User Story 3 - Review documented training evidence (Priority: P3)

As a reviewer, I want an English audit document linking the current behavior and measured results to source code, saved outputs, and run logs.

**Why this priority**: Persistent explanations make later comparisons possible without relying on recollection or terminal screenshots.

**Independent Test**: Follow the document's references and reconcile its fold facts, parameters, and metrics with saved evidence.

**Acceptance Scenarios**:

1. **Given** existing saved outputs, **When** they are documented, **Then** they are labeled as historical artifacts, not a newly executed or independently reproduced result.
2. **Given** an approved future instrumented run, **When** its results are documented, **Then** CV per-fold metrics, CV aggregate metrics, and final locked-test metrics remain separate and reference that run's evidence.
3. **Given** a code/documentation or selection-policy inconsistency, **When** it is found, **Then** the report records its evidence, implications, and deferred status without modifying the underlying behavior.
4. **Given** any reported R², **When** it is compared with 0.85, **Then** 0.85 is described only as a reference, never as a pass/fail gate or reason to alter folds or retrain repeatedly.

### Edge Cases

- Small datasets may produce fewer folds, adjusted block sizes, or no usable folds. Record the actual behavior; preserve existing failures rather than introducing a new splitting fallback.
- Remainder rows may enlarge the final validation block. Record actual sizes, not assumed 200-row counts.
- Input indices may be duplicated or non-contiguous. Membership evidence must remain unambiguous through dataset-relative row positions and dataset identity, without adding identifier features.
- Undefined/non-finite metrics must be explicitly labeled as unavailable/non-finite, not silently converted to zero or success.
- A tuning branch that is not selected must be marked skipped with its actual reason; do not execute it for logging coverage.
- Repeated invocations must not overwrite previous logs or duplicate terminal events through accumulated logging setup.
- If file logging cannot initialize or fails mid-run, emit an explicit English terminal warning and disclose incomplete evidence; do not silently claim a complete log or modify model decisions. Preserve the existing training/error flow.
- Do not dump raw records, credentials, environment secrets, or arbitrary user-supplied text into logs. Retain row references and numeric audit evidence instead.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The feature MUST be additive observability and documentation only. Existing dataset cleaning, feature policy, preprocessing, splits, seeds, estimator parameters, tuning order, selection logic, fit/predict call counts, and scientific output contracts MUST remain unchanged.
- **FR-002**: All newly added code identifiers, comments, docstrings, owned log messages, and feature documentation MUST use English. Existing unrelated code and comments MUST NOT be translated or refactored.
- **FR-003**: Each run MUST have a unique, non-overwriting `.logs` file, announced in the terminal, with matching training audit events in both destinations. Presentation-only progress animation need not be copied.
- **FR-004**: Each event MUST include timestamp, run identity, severity, step identity, operation, and status; model, trial, split, and fold context MUST be included where applicable. Completion events MUST include elapsed time.
- **FR-005**: Run evidence MUST identify the source input and content hash, relevant configuration, seed, feature/target contract, code version and dirty-worktree state, and runtime/dependency versions sufficient to contextualize reproducibility.
- **FR-006**: Logs MUST cover the existing salary-training path: development/test partitioning, fold construction/use, existing ablation and fold-importance fits, model comparison, each executed tuning candidate, actual selection, final development fit, locked-test prediction/scoring, and existing model/evaluation artifact writes and reload checks.
- **FR-007**: Fold evidence MUST include actual membership as dataset-relative positions, sorting policy, requested/effective split counts, block size, remainder handling, train/validation row counts and period bounds, and observed row/period overlap. It MUST be derived from partitions actually consumed, not a separately assumed split. Membership may be recorded once and referenced by later events within the same log.
- **FR-008**: Each executed candidate/fold MUST report actual estimator parameters and seed, fit and prediction durations, MAE, RMSE, R², and MedAE where computed. Aggregate evidence MUST identify its aggregation method and the actual ranking/selection basis. Additional descriptive summaries MUST NOT influence selection.
- **FR-009**: Logs MUST clearly distinguish CV, tuning, final-development fitting, and locked-test evaluation. Logging MUST NOT open the locked test earlier, introduce additional test evaluations, or drive model decisions from test scores.
- **FR-010**: Failed/cancelled runs MUST retain completed events and identify the last active operation and available error context while preserving existing exception and exit behavior. Skipped operations MUST not be presented as executed.
- **FR-011**: An English audit document MUST explain the current workflow, fold policy, actual selection rules, known discrepancies, log fields and location, reproduction instructions, and artifact-backed metrics. It MUST distinguish static inspection, historical artifacts, and newly verified runtime evidence.
- **FR-012**: R² = 0.85 MUST remain an informational reference only. No quality gate, tuning objective change, split change, new search, or repeated retraining may be introduced to reach it.
- **FR-013**: Any new user-controlled log destination or run identifier MUST be validated before file writes, reject unsafe traversal/overwrite, and yield a clear English warning on invalid input. No new configurable input is required; existing validation MUST remain intact.
- **FR-014**: Training MUST remain offline. No Streamlit changes, training triggered by reporting, new external logging service, or dependency upgrade is included.
- **FR-015**: Verification MUST demonstrate unchanged partitions, parameters, fit/predict call counts, selected model, predictions, and metrics on fixed inputs/seeds, allowing only explicitly justified numerical tolerances and expected differences in timing/log metadata.

### Constitutional Requirements *(mandatory)*

- **Data boundary**: Preserve `annual_salary_usd` as target, existing allowed/blocked features, development/locked-test cutoff, and train-partition-only fitted preprocessing. Record suspected boundary violations as findings; do not silently claim leakage safety from sorted row order alone.
- **Artifact contract**: Existing CSV/JSON/model bundle names, schemas, and consumers remain unchanged. New logs and audit documentation are additive and include provenance. The audit must not overwrite historical metrics to imply reproduction.
- **Streamlit boundary**: No UI change. Existing offline producers remain responsible for training; presentation remains an artifact consumer.
- **User input validation**: No new upload, widget, credential, or inference path. Any introduced log path/run ID is subject to FR-013. Secrets and raw-row payloads are excluded from new logs.
- **Scientific interpretation**: Report measured dataset-specific fit, not causal salary economics or production fitness. Keep CV and locked-test evidence separate, disclose existing test exposure, and preserve actual selection behavior without endorsing inconsistencies.
- **Documentation impact**: Document affected training operations, source locations, configuration, partition policy, trial/selection evidence, failure behavior, and log/output references. Reference existing diagnostic charts where available; do not create a new visualization workflow.
- **Verification evidence**: Before implementation, add failing-first logging acceptance tests and characterize existing behavior. Later compare before/after results on controlled fixtures and an approved offline integration run in an isolated output location. No training run is authorized by this specification-writing step.
- **Graphify/Karpathy review**: Existing graph queries identified the shared core training functions and compatibility facade. Graph locations are stale relative to current source and must be checked against source. Prefer surgical instrumentation; a documentation-only spec does not require rebuilding the graph.

**Existing governance discrepancy — not a requested correction**: Constitution VI.4 specifies lowest-mean-CV-MAE selection, while current manual Random Forest tuning ranks candidates by CV R². VI.2 also calls for candidate-level locked-test metrics, whereas the inspected final evaluation path evaluates the selected model. This feature documents rather than expands or repairs these behaviors. The plan must explicitly surface these inherited compliance gaps for maintainer disposition; it must not claim full compliance, amend the constitution, add test evaluations, or silently change selection rules. New logging must preserve leakage safety and evidence requirements.

### Key Entities

- **Training run**: One execution, identified by run ID, source hash, version/configuration/seed, timestamps, status, and evidence locations.
- **Training event**: A contextualized start, completion, skip, warning, failure, or cancellation for an existing operation.
- **Split/fold evidence**: Actual partition membership, ordering and period information linked to each consuming model or trial.
- **Candidate evaluation**: Parameters, seed, per-fold and aggregate metrics, durations, and actual selection decision.
- **Audit finding**: Observed behavior or inconsistency with source/artifact references, verification status, and deferred follow-up.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: For every executed operation in the agreed training coverage inventory, a reviewer can identify its start and completion or failure/cancellation in both terminal capture and the run log, with no missing model/trial/fold context.
- **SC-002**: Every logged fold's membership, counts, and periods reconcile with the partitions actually used; a reviewer can verify membership consistency across every compared candidate.
- **SC-003**: Controlled before/after verification shows unchanged scientific decisions, partitions, parameters, call counts, predictions, metrics, and existing output schemas, within documented numerical tolerance where needed.
- **SC-004**: All newly owned log messages and added code comments/docstrings/identifiers are English. Successful, failed, cancelled, repeated-run, and file-logging-failure scenarios are covered by acceptance checks.
- **SC-005**: Every numeric performance claim in the audit cites its source artifact/run and labels CV versus locked test and historical versus newly verified evidence. No acceptance check requires R² to exceed 0.85.

## Assumptions

- Intended users are the project maintainer and training-results reviewer, running the current local offline pipeline.
- “Do not modify current code” means no modeling behavior changes; the user explicitly permits minimal additions needed for logging and supporting tests/documentation.
- “Each training step” means observable operations in the current salary-training workflow, not new estimator internals such as per-tree/per-iteration callbacks, and not a redesign of segmentation or the UI.
- Logs include all computed training metrics and precise fold row references, not raw dataset dumps. Detailed fold membership is emitted once per split and referenced thereafter to avoid needless duplication.
- A `.logs` suffix means a plain-text, human-readable file per run, not a directory named `.logs`. The exact additive storage location will be settled in the implementation plan.
- Historical outputs are available but their provenance and reproducibility must be checked before comparison with any new run. Previously evaluated locked-test results cannot be described as newly unseen evidence.
- Approval to write this spec is not approval to implement, retrain, replace artifacts, install dependencies, or correct discovered issues. Plan/tasks and explicit implementation approval remain required.
