# Feature Specification: Review Branch B training and expose GridSearchCV evidence

**Feature Branch**: `feat/verify-008`  
**Created**: 2026-09-20  
**Status**: Approved and implemented for the offline training/validation producer; latest user scope excludes additional UI work.  
**Input**: Review the Branch B train/test split and leakage controls, replace the current manual Random Forest search with `GridSearchCV`, regenerate or invalidate affected evidence, and present clear fold, search, metrics, and training-step evidence on Pages 04–05. Page 06 may only be changed if its serving artifact contract changes.

## Scope amendment

The latest implementation request explicitly limits this pass to training and validation. Existing inherited Page 04/05 edits are preserved but not expanded or redesigned. The offline producer must emit complete English logs and immutable evidence that a UI can consume later. Preprocessing/data preparation and Page 06 remain unchanged. The requested 80/19/1 values are targets: strict chronology uses complete observed months, so both requested and actual counts/shares must be logged rather than forcing row-level percentages that would mix time periods.

## User Scenarios & Testing

### User Story 1 — Trust the revised offline evaluation (Priority: P1)

A reviewer can inspect a complete Branch B run and verify that chronological partitions, outer folds, nested tuning folds, and the locked-test policy prevent target/future leakage.

**Independent test**: Construct ordered fixture data and prove that every train period precedes its validation period, preprocessing is fit inside each fold, GridSearchCV sees only the relevant inner-development rows, and no holdout/reserve row occurs in fit or search input.

**Acceptance scenarios**:

1. **Given** ordered training data, **when** the split/folds are created, **then** each outer and inner fold is chronological, disjoint, and records exact row/month counts.
2. **Given** a selected Random Forest family, **when** tuning occurs, **then** `GridSearchCV` evaluates a declared `n_estimators` × `max_depth` grid with the matching inner temporal folds and ranks candidates by declared CV metric.
3. **Given** any known-exposed locked-test/reserve population, **when** a new evaluation would access it, **then** the run fails closed and does not claim a new untouched-test result.

---

### User Story 2 — Review training decisions on Pages 04–05 (Priority: P1)

A stakeholder can read chart-first summaries and expand evidence at the end of Page 04 or Page 05 to see partition/fold construction, GridSearchCV candidates, R²/MAE/RMSE/MedAE, runtime/log records, and a step-by-step training transcript.

**Independent test**: Load a valid regenerated fixture pack through the Streamlit entrypoint and verify Page 04 shows outer/inner split evidence and Page 05 shows GridSearchCV grid/folds/winner/final evidence, without fitting or loading models in the UI.

**Acceptance scenarios**:

1. **Given** a valid revised evidence pack, **when** Page 04 is opened, **then** it shows fold counts, chronology/exclusion checks, candidate metrics including R², and an evidence-derived step log in a collapsed end-of-page detail section.
2. **Given** a valid revised evidence pack, **when** Page 05 is opened, **then** it shows GridSearchCV scoring, parameter grid, inner-fold split, all candidate results, winner parameters, and final evaluation labels in a collapsed end-of-page detail section.
3. **Given** unavailable, invalid, or deliberately invalidated evidence, **when** either page opens, **then** it gives a bounded unavailable/invalidation explanation and never substitutes old metrics as revised evidence.

---

### User Story 3 — Preserve safe serving (Priority: P1)

A Page 06 user continues to receive predictions only from a validated compatible serving bundle, and can identify whether its bundle predates the revised training contract.

**Independent test**: Verify compatible serving continues unchanged; verify a changed model metadata/schema is rejected with an actionable message rather than silently loading an incompatible bundle.

**Acceptance scenarios**:

1. **Given** the serving contract is unchanged, **when** the revised training evidence is produced, **then** Page 06 behavior and predictions remain unchanged.
2. **Given** the serving bundle contract changes, **when** Page 06 opens, **then** it validates the revised provenance/version before loading; an old bundle fails closed.

### Edge Cases

- The current dataset’s historical locked-test/reserve rows are already exposed by committed artifacts; a new supposedly pristine test evaluation is prohibited.
- `GridSearchCV` grid candidates, folds, scoring direction, or seed are missing, inconsistent, or non-finite.
- A parameter candidate fails in one fold; its error/status is persisted and it cannot win silently.
- Outer/inner folds overlap, include later months in training, use fewer than required observed months, or contain preprocessing fit outside the fold pipeline.
- Regenerated files have a changed schema, checksum mismatch, or an old UI pack remains beside a revised pack.
- Page 04/05 logs are too large: previews are bounded, source downloads remain complete, and derived rows are labelled as derived rather than raw telemetry.

## Requirements

### Functional Requirements

- **FR-001**: The offline Branch B training path MUST use `GridSearchCV` for Random Forest tuning; the grid MUST include `n_estimators` and `max_depth` and record all declared parameter values, scoring, fold identities, seed, ranks, mean/std metrics, fit time, and failures.
- **FR-002**: The GridSearchCV splitter MUST consume only chronological inner-development folds associated with each outer fold/final development context; outer validation, evaluation holdout, and inference reserve rows MUST not appear in its fitting/search inputs.
- **FR-003**: Candidate family selection MUST remain based on outer chronological CV evidence before tuning/final evaluation. GridSearchCV MUST tune only the frozen eligible family and MUST not select a family using the held-out test.
- **FR-004**: The split policy and fold builders MUST validate chronology, row disjointness, train-only preprocessing, target/identifier exclusion, and exact partition/fold count reconciliation; violations MUST stop the run with a stable reason code.
- **FR-005**: The producer MUST emit a versioned evidence contract that captures split declarations, outer/inner fold membership/counts, complete GridSearchCV results, selected parameters, R²/MAE/RMSE/MedAE, raw structured events, and an evidence-derived method transcript.
- **FR-006**: Any change to training code, output schema, or consumer contract MUST invalidate existing affected training and UI evidence. Revised evidence may be generated only from a documented eligible, unexposed evaluation population with a fresh approval; otherwise it MUST remain invalidated/unavailable.
- **FR-007**: Pages 04 and 05 MUST present a visible `Latest pipeline training validation` summary sourced from the active Branch B pipeline evidence, followed by chart-first report summaries; detailed split/log/evidence sections MUST be native collapsed expanders located after the established analytical content. The visible summary MUST state when its locked-test evidence is historical/exposed.
- **FR-007a**: The separate strict future-reserve `training-validation/v1` section MUST remain clearly labelled as unavailable when the current workspace is exposure-blocked. It MUST not hide, replace, or prevent the visible latest-pipeline evidence.
- **FR-008**: Page 04 MUST show the full temporal split and every outer/inner fold’s train/validation periods and row counts, exclusions, chronology/overlap checks, candidate metrics including R², and model decisions.
- **FR-009**: Page 05 MUST show the GridSearchCV scoring rule, parameter grid, chronological inner-fold allocation, candidate rank/mean/std metrics, winning `n_estimators`/`max_depth`, final-fit scope, and clearly labelled CV versus historical/new holdout metrics.
- **FR-010**: Pages 04–05 MUST show bounded raw event previews plus an explicitly labelled evidence-derived step-by-step training transcript and complete source/download evidence. The UI MUST not train, tune, predict, publish, or deserialize training-validation bundles.
- **FR-011**: Page 06 MUST remain unchanged unless a revised serving artifact schema/provenance requires a surgical compatibility check; it MUST never load a mismatched bundle silently.
- **FR-012**: Training logs, artifact validation, and UI unavailable states MUST use stable, actionable messages with run/provenance identifiers and must not expose raw exceptions, paths outside the workspace, or credentials.
- **FR-013**: Documentation MUST update the Branch B training chain, split/leakage review, GridSearchCV methodology, artifact version/invalidation decision, and UI evidence interpretation.

### Constitutional Requirements

- **Data boundary**: Target is `annual_salary_usd`. Branch B uses the declared raw salary features only; identifiers and target-derived/post-outcome fields are excluded. All transformers fit within a fold-local pipeline. Chronological development folds select/tune; a genuinely unseen evaluation holdout is opened only after all decisions freeze. The current historically exposed population cannot be reclassified as pristine.
- **Artifact contract**: Replacing manual search changes producer outputs and Page 04/05 consumer evidence. Existing affected packs and active UI evidence must be invalidated or regenerated. Page 06 bundles are untouched unless their schema/provenance contract changes.
- **Streamlit boundary**: Streamlit only validates and renders immutable evidence; offline CLI owns splitting, GridSearchCV, fitting, events, and publishing.
- **User input validation**: There are no new free-text UI inputs. Offline workspace, approval, policy, data schema, dates, grid values, paths, IDs, hashes, and exposure attestation are validated before use.
- **Scientific interpretation**: CV, historical test, and any new eligible holdout results remain separately labelled. R² is descriptive fit evidence, not causal or production evidence. q90 remains empirical and non-calibrated unless separately validated.
- **Documentation impact**: Update `docs/BRANCH_B_COMBINATION.md`, `docs/TRAINING_VALIDATION.md`, `docs/MODEL_UI.md`, and affected contracts/quickstart with changed method and invalidation status.
- **Verification evidence**: Failing-first split/leakage/GridSearchCV/artifact-schema tests; producer integration with a synthetic eligible fixture; Page 04/05 AppTests; Page 06 compatibility test if changed; targeted/full pytest, Ruff, `git diff --check`, artifact hashes, and browser review.
- **Graphify/Karpathy review**: Existing graph query identified `training_partitions.py`, `training_search.py`, `training_validation.py`, Pages 04–06, and presentation readers. The simplest viable design replaces only the search implementation and its evidence projection; no UI training controls or duplicated pipeline are introduced.

### Key Entities

- **TemporalSplitEvidence**: Versioned partition boundaries, row IDs/counts, month ranges, exclusion and exposure status.
- **GridSearchEvidence**: Search context, parameter grid, scorer, inner-fold identities, candidate-level mean/std/rank/timing/status, and selected parameters.
- **TrainingStepEvidence**: Ordered raw event or explicitly derived training step with source reference and status.
- **EvidenceCompatibilityState**: Valid, invalidated, unavailable, or incompatible status with safe reason and required next action.

## Success Criteria

- **SC-001**: Tests prove every GridSearchCV fit uses only the intended temporal inner-development rows and that outer validation/holdout/reserve targets are inaccessible during tuning.
- **SC-002**: Tests prove the persisted results contain every declared `n_estimators` × `max_depth` candidate, deterministic rank/winner, finite R²/MAE/RMSE/MedAE where applicable, and source-linked event/step evidence.
- **SC-003**: With a valid fixture pack, Page 04 and Page 05 identify every fold’s periods/counts and GridSearchCV candidate/winner evidence without calling training, prediction, publication, or model deserialization.
- **SC-004**: With no eligible regenerated data, current affected evidence is visibly invalidated/unavailable rather than presented as results of the revised logic.
- **SC-005**: Page 06 compatibility/prediction regression tests pass unchanged unless a documented serving-contract migration is required.
- **SC-006**: Targeted tests, full pytest, Ruff, `git diff --check`, and required browser checks pass; regenerated artifacts are validated against the revised contract only after eligible-data approval.

## Assumptions

- The user’s authorization permits changes to offline training and Pages 04–06, but does not waive the constitution’s leakage and evidence rules.
- The review finding is confirmed: `src/ai_job_market/training_search.py` currently uses manual sequential sweeps, not `GridSearchCV`.
- The current known-exposed historical test/reserve cannot honestly support a fresh post-change locked-test result; fresh eligible data and a new approval are required for such a result.
- Page 06 does not require UI changes if its serving artifact contract remains compatible.
