# Specification Quality Checklist: Training-Only Split and Validation Evidence

**Purpose**: Validate clarified scope and planning artifacts before task approval.  
**Created/Reviewed**: 2026-09-19  
**Feature**: [spec.md](../spec.md)  
**Status**: Implementation approved and code/fixture verification complete; real-data scientific and human-reader acceptance remain gated.

## Content Quality

- [x] Focused on reviewer value, training correctness and explicit data boundaries.
- [x] Plain-English stories, 5W1H and translated Vietnamese feedback included.
- [x] All mandatory specification sections completed.
- [x] Specification states behavior; technical implementation details live in plan/design contracts.

## Requirement Completeness

- [x] All three clarification questions resolved and recorded with user decisions.
- [x] 19% is evaluation holdout, not tuning data; 1% is inference-only, never evaluated.
- [x] Whole-month ordering is strict across all three partitions; exact percentages are subordinate.
- [x] Classification scope is educational only, no classifiers or invented labels.
- [x] Success criteria measurable, technology-independent and acceptance scenarios defined.
- [x] Edge cases include chronology, insufficient nested history, undefined metrics, exposure and failures.
- [x] Scope excludes UI, upstream preprocessing/preparation and deployment infrastructure.
- [x] Assumptions/dependencies disclosed, including proposed fixed category/experience pair.
- [x] Historical test data is not falsely described as an unseen reserve or pristine evaluation.

## Planning Readiness

- [x] `plan.md`, `research.md`, `data-model.md`, `quickstart.md` and `contracts/training-evidence.md` exist.
- [x] Five models, five expanding monthly folds, metrics/runtime/ablation/importance/drift covered.
- [x] Bounded nested RF tuning, actual carried settings, one-time evaluation and residual/q90 evidence defined.
- [x] Existing artifacts remain historical; new namespace/reuse/invalidation policy explicit.
- [x] Tasks use exact paths, dependencies, tests-first checkpoints and story traceability.
- [x] Constitution reviewed pre/post design; no leakage waiver or silent legacy experiment replacement.
- [x] Runtime review adds controlled prediction latency, throughput, cost/size/context, separate quality/speed findings and non-defaulting operational budgets.
- [x] Human transcript, agent summary, metric catalog and typed future-UI snapshot derive from the same evidence, with parity/error/progress tests.
- [x] Extra regression bias/tail/baseline/gap metrics have formulas, units, scopes and null handling in `contracts/runtime-observability.md`.
- [x] No benchmarks were executed or fastest-model/optimization claims invented during review.
- [x] Fold-method explanation now explicitly requires exact rows/months, nested parents, added history, percentage denominators and monthly-total/membership reconciliation in human/agent/UI-ready evidence.
- [x] Every candidate and final full/Top-2 role requires a scoped conclusion/status; non-selected/failed models cannot disappear or receive fabricated favorable verdicts.
- [x] Existing partition/report/parity/walkthrough tasks cover the amendment; no extra fits or UI implementation are introduced.
- [x] Dependency-ordered 51-task package retained with implementation/data/human gates.
- [x] User explicitly approved implementation on 2026-09-19 after the runtime/fold/conclusion amendments.

## Execution Readiness (not a specification defect)

- [ ] Eligible unseen future inference-reserve data available — not established; current latest month is historically exposed.
- [ ] Actual dataset-specific whole-month counts/ratios and exposure provenance approved — required before a real run.
- [ ] Implementation tests and real-data validation executed — 183 tests pass (1 skipped), but real-data validation is blocked and was not executed.
- [ ] Two independent readers completed comprehension acceptance — pending; automated parity tests are not a substitute.

## Notes

Implementation adds only the isolated training-validation producer, contracts, tests and documentation. Controlled model fitting/prediction occurred only in temporary test workspaces; no real-data training pack, preprocessing/preparation output, existing model artifact, UI source, dependency file or serving pointer changed. Existing untracked `uv.lock` is preserved. Branch number `006` and spec directory number `004` are independent under repository conventions.

`AGENTS.md` now references the new plan. Assessment evidence remains at `.specify/assessments/validate-training-split/research.md`. Unknown/known-exposed reserve provenance blocks full execution; do not check that box by relabelling old data.
