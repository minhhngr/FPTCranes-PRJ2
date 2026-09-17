<!--
Sync Impact Report
- Version change: 1.1.1 -> 1.2.0
- Modified principles:
  - IV. Offline Training, Read-Only Reporting -> IV. Offline Training, Streamlit Presentation Boundary
  - V. Honest Interpretation and Secure Defaults -> V. Honest Interpretation and Universal Input Validation
  - VI. Model Evaluation Documentation -> VI. Model Evaluation Documentation
- Added principles:
  - VII. Graphify-Guided, Karpathy-Simple Changes
- Added sections: none
- Removed sections: none
- Templates requiring updates:
  - ✅ .specify/templates/plan-template.md
  - ✅ .specify/templates/spec-template.md
  - ✅ .specify/templates/tasks-template.md
  - ✅ .specify/templates/commands/*.md (directory absent; no command templates to update)
  - ✅ README.md (reviewed; already aligned with offline Streamlit artifact boundary)
  - ✅ AGENTS.md (reviewed; no stale constitution reference)
- Follow-up TODOs: none
-->
# AI Job Market Salary Prediction Constitution

## Core Principles

### I. Data Integrity and Leakage Safety

Every model change MUST declare the target, feature policy, development period, and locked
test period before target-aware analysis begins. Target values, target-derived fields,
identifiers, and post-outcome information MUST NOT enter model features. Preprocessors,
vocabularies, imputers, scalers, and feature selection MUST be fitted only on the applicable
training partition. Model selection and tuning MUST use development-period temporal
validation; the locked test MAY be opened only after selection is complete. These rules
protect reported performance from leakage and preserve the meaning of an unseen test.

### II. Reproducible Staged Artifacts

The documented staged workflow MUST remain reproducible from the versioned source dataset
and pinned project dependencies. Stochastic operations MUST use an explicit seed. Each run
MUST emit inspectable intermediate evidence, final metrics, model metadata, and a serialized
inference bundle whose feature schema matches serving. Data provenance MUST include the input
path or identifier and a content hash. A change to an artifact name, schema, stage meaning, or
consumer contract MUST update every producer, consumer, test, and relevant document in the
same change.

### III. Evidence-First Verification (NON-NEGOTIABLE)

Behavior changes MUST begin with an executable test or validation criterion that fails for
the missing or incorrect behavior, followed by the smallest implementation that makes it
pass. Data and model changes MUST test leakage gates, split boundaries, schemas,
deterministic behavior, and metric calculations as applicable. Artifact or Streamlit changes
MUST include an integration test of the producer-consumer contract. Before merge, the
relevant pytest suite MUST pass and the documented pipeline and application entrypoints MUST
be exercised in proportion to the change. Printed values without assertions are not
sufficient evidence.

### IV. Offline Training, Streamlit Presentation Boundary

Model fitting, tuning, and artifact generation MUST occur in the offline pipeline, never in
the Streamlit process. Streamlit MUST remain a presentation and orchestration layer: it MAY
load data, validate inputs, select approved artifacts, render charts, and display inference
results, but it MUST NOT refit models or mutate training artifacts during interaction. Core
business logic, data processing, model training, and inference contracts MUST live in
importable Python modules that can be tested without launching Streamlit. Missing,
incompatible, or corrupt artifacts MUST produce a clear actionable error rather than silent
recomputation or fabricated defaults.

### V. Honest Interpretation and Universal Input Validation

Reports and interfaces MUST distinguish descriptive correlation, fitted-model importance,
validation performance, locked-test performance, and causal claims. No result from this
dataset MAY be presented as causal salary economics or production fitness without external
validation. Prediction intervals MUST state their empirical basis and limitations.

Every user-controlled input MUST be validated before use, including uploaded files, sidebar
controls, form values, URL/query parameters, credentials, selected run IDs, file paths, and
model-prediction scenarios. Validation MUST define allowed schema, type, range, enum, size,
path, authorization, and temporal constraints where applicable. Invalid input MUST fail
closed with a clear user-facing message and MUST NOT trigger training, inference, file
writes, artifact selection, or path traversal. Demo credentials MUST be identified as
local-only; shared deployments MUST obtain credential material from secrets or environment
configuration and MUST NOT commit plaintext secrets.

### VI. Model Evaluation Documentation

Every model evaluation deliverable MUST document the complete chain from train-test split
through final selection with traceable, artifact-backed evidence. Production code, public
functions, pipeline stages, feature contracts, validation rules, and model artifacts MUST be
documented with enough detail for a future maintainer to understand purpose, inputs,
outputs, assumptions, and failure modes. The following sub-requirements are mandatory:

1. **Split specification.** The temporal split identity MUST be stated. The split MUST be
   performed on the raw cleaned DataFrame, not on an encoded correlation view.
   Preprocessing MUST be fitted inside the pipeline, never before the split.
2. **Candidate ladder.** At minimum five models (Dummy median floor, Linear Regression,
   Ridge, Random Forest, Gradient Boosting) plus optional SVR MUST be trained on identical
   splits. Each candidate MUST report MAE, RMSE, R², and MedAE on both temporal CV folds
   and the locked test.
3. **Hyperparameter tuning.** Tuning MUST be nested inside development temporal folds only;
   the locked test MUST NOT be touched until all selection decisions are frozen. Every trial
   MUST log parameters, fold IDs, seed, and metrics.
4. **Best model selection.** The winner MUST be selected by lowest mean temporal CV MAE
   before the locked test is opened. When candidates overlap within fold variance, the
   simpler model MUST be preferred.
5. **Feature importance.** Both encoded-feature importance from the estimator and
   raw-feature-family permutation importance on the locked test MUST be reported.
6. **Interpretation caveats.** Documentation MUST state known contradictory or
   likely-synthetic signals, that importance values are model reliance rather than causal
   evidence, that R² is a fit metric for this dataset, and that model under-performance must
   be explained from evidence rather than assertion.
7. **Uncertainty communication.** Every prediction summary MUST include a practical interval
   or MedAE rather than bare point estimates.
8. **Artifact traceability.** Results MUST cite the specific output files and MUST be
   reproducible from the versioned source dataset, pinned dependencies, and recorded seeds.
9. **Visualization.** Documentation MUST embed or reference generated charts for model
   comparison, locked-test diagnostics, and feature importance. Charts MUST be referenced
   from pipeline output directories, not manually recreated.

### VII. Graphify-Guided, Karpathy-Simple Changes

Before implementing non-trivial code changes, contributors MUST inspect the repository with
Graphify or an existing Graphify graph to understand affected files, dependencies, and risks.
Changes MUST follow Karpathy-style discipline: state assumptions, prefer the simplest working
design, make surgical edits, avoid speculative abstractions, and define verifiable success
criteria before coding. Code changes MUST update the Graphify graph or record why a scan-only
review was sufficient.

## Scientific and Operational Constraints

- Python, the offline pipeline, generated file artifacts, and Streamlit are the established
  stack. New infrastructure requires a documented need and a simpler alternative analysis.
- The source dataset is an academic snapshot with contradictory and synthetic-looking
  structure. Results are suitable for supervised-regression study and controlled scenarios,
  not unqualified labor-market decisions.
- Output packs, saved model bundles, and metadata files form a published internal contract.
  Compatibility changes MUST be explicit and tested.
- Generated evidence MUST retain units, partition names, model-selection basis, and enough
  provenance for a reviewer to trace a displayed claim back to its source artifact.
- Runtime failures MUST identify the missing or invalid input and the command needed to
  regenerate it. Silent fallbacks that alter scientific meaning are prohibited.

## Development Workflow and Quality Gates

1. Specifications MUST identify affected data boundaries, artifact contracts, scientific
   claims, user-controlled inputs, security considerations, documentation impact, and
   measurable acceptance evidence.
2. Plans MUST pass every Constitution Check before research and again after design. Any
   exception MUST be recorded in Complexity Tracking with the need and rejected simpler
   alternative; exceptions cannot waive leakage safety, input validation, or evidence
   requirements.
3. Tasks MUST put tests and validation before implementation, retain traceability to user
   stories, include documentation work, and include Graphify review or graph-refresh work
   when interfaces or code change.
4. Reviews MUST inspect the actual diff, run relevant automated tests, verify the exact
   pipeline or Streamlit entrypoint affected by the change, and reject model fitting in UI
   paths or unvalidated input paths. Model-result reviews MUST cite generated artifacts
   rather than recollection or console-only output.
5. Code changes MUST run `graphify update .` after verification unless the change records a
   valid reason a scan-only or query-only Graphify review was sufficient. Documentation-only
   governance changes MAY skip graph updates when they do not alter indexed source
   relationships.

## Governance

This constitution supersedes conflicting project practices and generated template guidance.
An amendment MUST be proposed as a documented diff, state its motivation and migration impact,
update dependent templates and guidance in the same change, and receive maintainer approval.

Versions follow semantic versioning: MAJOR for incompatible principle removals or
redefinitions, MINOR for new principles or materially expanded obligations, and PATCH for
non-semantic clarification. The Sync Impact Report at the top of this file MUST record each
amendment and any deferred follow-up.

Every feature specification, implementation plan, task list, and code review MUST verify
constitutional compliance. Reviewers MUST reject unexplained violations. Complexity is
acceptable only when its necessity and simpler rejected alternative are documented. Runtime
commands and project-specific guidance remain in `README.md` and `AGENTS.md`, but neither may
override this constitution.

**Version**: 1.2.0 | **Ratified**: 2026-08-19 | **Last Amended**: 2026-09-17
