# Specification Quality Checklist: Model Comparison, Diagnostics and Prediction UI

**Purpose**: Validate specification completeness and quality before planning  
**Created**: 2026-09-19  
**Updated**: 2026-09-19 after table consistency and training-method explanation research  
**Feature**: [spec.md](../spec.md)

## Content Quality

- [x] CHK001 Product requirements specify observable behavior; the explicit user architectural constraint and mandated constitutional references are recorded separately from UI intent.
- [x] CHK002 Focused on reviewer, auditor, scenario-user and evidence-maintainer value.
- [x] CHK003 User journeys and limitations are understandable to stakeholders; model/metric terminology matches the requested domain.
- [x] CHK004 All mandatory template sections completed, with offline evidence generation added as US4.

## Requirement Completeness

- [x] CHK005 No clarification markers remain; Q1–Q3 dispositions recorded from the user's response.
- [x] CHK006 Requirements are testable; FR-015 defines offline generation, FR-016 observed DEV pairs, and FR-017 strict defaults with bounded acknowledged exceptions.
- [x] CHK007 Success criteria cover measurable reconciliation, layout, validation, queue, evidence generation/reuse and walkthrough outcomes.
- [x] CHK008 Success criteria describe observable results rather than prescribing UI APIs.
- [x] CHK009 Acceptance scenarios cover comparison, diagnostics, inference, exports, stale results, missing evidence and offline publication/reuse.
- [x] CHK010 Edge cases cover invalid metrics/packs, unknown actuals, fractional bounds, invalid pairs, queue limits and exception labels.
- [x] CHK011 Scope is bounded: supplemental offline generation and Pages 04–06; unchanged god module/primary pipeline/data/split, no fresh tuning or pristine-data claims.
- [x] CHK012 Dependencies, proposed defaults, provenance limitations and architectural constraints are explicit.

## Feature Readiness

- [x] CHK013 Functional requirements map to acceptance scenarios and verifiable success criteria.
- [x] CHK014 Seven user stories cover the primary journeys, including generation prerequisites, visual/readability remediation and training-method auditability.
- [x] CHK015 Dependencies have explicit dispositions: generate missing supplemental evidence; verify/reuse baseline evidence; mark unavailable optional history; do not manufacture external/pristine data.
- [x] CHK016 Product spec does not choose new infrastructure/dependencies; the user's new-module constraint is preserved and technical design resides in the plan/contracts.
- [x] CHK017 Evidence honesty, no new training in UI, historical test exposure and inherited governance limitations are explicit.
- [x] CHK018 Planning approval is distinguished from implementation approval; generated tasks remain unchecked until separately approved and executed.
- [x] CHK019 Semantic colors are defined by evidence role, include redundant non-color encoding and explicitly make historical-test error a red column series.
- [x] CHK020 Chart P1/P2/P3 width rules are testable, responsive and include a label-readability promotion rule rather than forcing every chart full width.
- [x] CHK021 Visual fidelity to the stakeholder input is explicitly separated from unsupported numeric, pristine-data and causal claims.
- [x] CHK022 The color/priority amendment changes presentation only and does not authorize model changes, new dependencies or custom CSS.
- [x] CHK023 Stakeholder overlap/unreadable-number feedback is translated into measurable geometry, label-density and browser acceptance requirements rather than a vague beautification request.
- [x] CHK024 Page briefs, metric glossaries and section questions define a novice reading path without adding long prose before every visual.
- [x] CHK025 Conclusion requirements are deterministic, source-bound and testable across changed rankings, missing evidence and queue mutations.
- [x] CHK026 Page-specific minimum conclusions answer winner/generalization/uncertainty/scenario questions while retaining scientific limitations.
- [x] CHK027 Figure profiles cover margins, height, automargin, tick formatting, long categories, direct-label density and dense scatter behavior.
- [x] CHK028 Table requirements cover reader-facing names/units and full machine-readable download preservation.
- [x] CHK029 AppTest is explicitly insufficient for pixel acceptance; browser screenshots, bounding boxes, console and accessibility review remain required.
- [x] CHK030 Scope remains Pages 04–06 and shared presentation helpers; no metric/model/policy/inference/export logic or dependency change is authorized.
- [x] CHK031 Table emphasis has observable semantics: bold is governing/selected evidence, italics are reference/context, and wide dataframes never display raw Markdown.
- [x] CHK032 Table names, units, precision, tie handling and highlight rules are consistent across Page 04–05 KPIs, charts and table tiers.
- [x] CHK033 The temporal-validation guide describes the actual adjacent sliding-window implementation, fold-local preprocessing, purpose, metrics and limitations rather than generic K-fold behavior.
- [x] CHK034 The tuning guide describes the actual anchor/coordinate stages, R² ranking, carried/fixed values, saved-selection source and non-nested limitation without unsupported rationale claims.
- [x] CHK035 Active supplemental evidence remains authoritative; optional historical audit logs have explicit completeness, source, path and checksum compatibility gates.
- [x] CHK036 Required fold/tuning evidence downloads and optional audit trace exports are specified without authorizing UI training or evidence regeneration.

## Notes

- **Validation result: 36 passed, 0 open product clarifications.** These are specification-quality checks, not feature test results or proof of scientific performance.
- User approved creating tasks for offline generation in new `.py` files, not adding logic to `src/ai_job_market/core.py`.
- Uniform experience bounds are the default; exceptions are explicit, experience-only, bounded and cannot bypass observed pairs/model validation. Observed mappings use DEV, not curated semantics.
- Available data cannot become pristine retrospectively. Generated examples are historical benchmark records with unchanged scoring inclusion, as documented in the revised spec and plan.
- Plan review must retain bounded inherited non-nested tuning/selection and global upload-boundary issues; no leakage or honesty requirement is waived.
- The baseline and explicitly approved T048–T055 presentation amendment were implemented and automatically verified. The shared figure-builder fingerprint changed, so the supplemental pack was versioned and regenerated; an unchanged rerun was then reused.
- The stakeholder explicitly approved T056–T067. Automated/source tasks T056–T064 and T066–T067 are complete. Chrome DevTools MCP is unavailable in the current harness, so T065 remains open and is not marked complete based on AppTest alone.
- The stakeholder explicitly approved T068–T079 with training only if needed. Existing active evidence and complete audit logs were sufficient; T068–T078 and automated T079 review work completed with no training/regeneration. Human/browser portions remain open.
