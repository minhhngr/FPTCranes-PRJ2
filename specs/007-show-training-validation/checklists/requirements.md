# Specification Quality Checklist: Show Offline Training Validation on Pages 04–05

**Purpose**: Confirm the UI integration is reviewable, fail-closed, and bounded before planning.  
**Created**: 2026-09-20  
**Feature**: [spec.md](../spec.md)  
**Status**: Original implementation and approved historical Branch B 5W1H amendment complete; browser and independent reader acceptance remain pending.

## Content Quality

- [x] User value is stated without prescribing implementation internals in user stories.
- [x] Pages 04 and 05 have independently testable journeys.
- [x] The confirmed unavailable behavior is a first-class P1 journey.
- [x] No unresolved clarification marker remains.
- [x] Edge cases cover missing, staging, corrupt, incompatible, multiple and partial packs.

## Scope and Safety

- [x] Streamlit is explicitly read-only and cannot trigger offline operations.
- [x] Existing supplemental evidence remains independent and is not blended.
- [x] No fixture, historical, hardcoded or old-output fallback is permitted.
- [x] Page 06, preprocessing, source data, pipeline, bundles, deployment and serving are excluded.
- [x] Existing complete artifacts are reused; no UI-only retraining/regeneration is allowed.
- [x] CV, holdout, inference reserve, historical test, scientific outcome and operational status remain distinct.

## Requirement Completeness

- [x] Latest-pack selection and deterministic tie behavior are defined.
- [x] Validation, path containment, schema, checksum, size and no-deserialization requirements are defined.
- [x] Page 04 fold/candidate/runtime/conclusion content is explicit.
- [x] Page 05 tuning/variant/holdout/importance/subgroup/uncertainty content is explicit.
- [x] Detailed outer/inner fold, 25 candidate-fold, tuning/variant/final step transcript requirements are explicit.
- [x] Raw human log, structured events and evidence-derived transcript are distinguished.
- [x] Five original log/report/summary downloads and complete generated transcript download are mandatory.
- [x] Table bounds, labels, units, precision and downloads are required.
- [x] Error messages, remediation and no-fallback behavior are measurable.
- [x] Existing-page independence and early-return placement are covered.
- [x] Documentation and Graphify/Karpathy boundaries are stated.

## Acceptance Readiness

- [x] Pure loader/view tests and real-entrypoint AppTest are required.
- [x] No-fit/tune/predict/load/publication spies are required.
- [x] Full regression, Ruff, diff and protected-hash checks are required.
- [ ] Two-reader comprehension acceptance completed — implementation-stage human acceptance.
- [ ] Browser visual acceptance completed — implementation-stage, pending tooling availability.
- [x] User approved original implementation tasks.
- [x] User approved amended Phase 9 historical Branch B 5W1H tasks T039–T048.

## Notes

The current real workspace has no eligible training-validation pack. The expected initial UI state is therefore a truthful unavailable expander. Synthetic fixture packs may be used only in tests and must never become runtime fallback data.
