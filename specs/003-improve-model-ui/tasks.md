# Tasks: Evidence-Backed Model UI and Offline Supplemental Artifacts

**Input**: `specs/003-improve-model-ui/`  
**Prerequisites**: [spec.md](spec.md), [plan.md](plan.md), [research.md](research.md), [data-model.md](data-model.md), [evidence contract](contracts/evidence-pack.md), [UI/scenario contract](contracts/scenario-ui.md), [quickstart.md](quickstart.md)  
**Status**: 76 of 79 tasks completed. The approved T068–T078 table/training-explanation work is complete. T044 remains open for independent human walkthroughs, T065 remains open for Chrome DevTools pixel/accessibility acceptance, and T079 remains open only for those unavailable browser/human acceptance portions; its code review, protected-hash and Graphify work is complete.

**Tests**: Failing-first tests are mandatory for changed behavior. Record red/green evidence in `specs/003-improve-model-ui/verification.md` during approved implementation. All training/generation tests run in isolated workspaces; real missing/invalidated supplemental artifacts are generated only at T040. No root main-pipeline rerun is part of this feature.

**Hard constraints**: New generation logic in new `.py` modules; `src/ai_job_market/core.py`, `pipeline.py`, baseline artifacts, data/config/lock and primary model selection remain unchanged. Default domain validation applies uniformly. Exceptions are explicit and experience-only; observed pairs are DEV-derived. Historical examples are not pristine.

## Phase 1: Setup and preservation

**Goal**: Confirm approval and establish a trustworthy working-tree baseline without changing source/artifacts.

- [X] T001 Record explicit implementation approval and protected-file SHA-256 inventory for `src/ai_job_market/core.py`, `pipeline.py`, source/config/lock and baseline output/artifact packs in `specs/003-improve-model-ui/verification.md`; preserve pre-existing untracked files and verify no baseline mutation before proceeding.
- [X] T002 Re-query Graphify for metadata/splits/serving relationships, verify current helper and dependency assumptions against `src/ai_job_market/core.py` and `streamlit.py`, and record differences or a confirmed unchanged plan in `specs/003-improve-model-ui/verification.md`.
- [X] T003 Build reusable tiny DEV/test, metadata, malformed-pack and counting-estimator fixtures in `tests/test_ui_evidence_contract.py` (extract a test-only fixture module if shared); freeze expected identities and assert fixtures never write to baseline `outputs/` or `artifacts/`.

**Checkpoint**: Approval is recorded, baseline evidence protected, and fixture identities documented. If current source/contracts differ materially from the plan, revise the plan before coding.

## Phase 2: Foundational contracts and policy

**Goal**: Establish importable integrity/policy primitives and minimal pure figure builders before producer/UI work.

- [X] T004 Add failing manifest/schema/path/hash/feature-order tests in `tests/test_ui_evidence_contract.py`, including symlink/escape refusal, incomplete publication, immutable references, legacy-origin limitations and component dependency fingerprints from `contracts/evidence-pack.md`.
- [X] T005 Implement versioned records, safe workspace paths, hashes, manifest/pointer loading and dependency-aware component validation in new `src/ai_job_market/ui_evidence_io.py`; pass T004 without importing or invoking a training orchestrator from the reader.
- [X] T006 Add failing observed-DEV pair, title-bound/median, support-count, target-independence and exception-matrix tests in `tests/test_scenario_policy.py`; cover fractional/default/single-value bounds, unknown pairs, nonfinite/bool years, immutable benchmark lookup and expired acknowledgement.
- [X] T007 Implement DEV-only policy construction and shared strict/experience-exception validation in new `src/ai_job_market/scenario_policy.py`; pass T006 with no target-derived statistics or test-informed policy and no manual-scenario bypass.
- [X] T008 Add failing pure chart/provenance tests in `tests/test_model_evidence_views.py` for three canonical output figures (candidate comparison, full-test actual/predicted, raw importance), explicit units and invalid/missing input states; assert figure data reflects source identities rather than document literals.
- [X] T009 Implement minimal pure figure/table builders and workspace/evidence-keyed read caches in new `src/pages/model_evidence.py`; pass T008 and ensure the offline producer can save figures without launching Streamlit or executing page/render code. Modernize only the touched `show_plot` width argument in `src/pages/common.py` and verify compatibility with existing consumers.

**Checkpoint**: Contract/policy tests pass; primitive figures can be built from fixtures; no model fit occurs during imports, loading or policy/view operations. T004→T005, T006→T007 and T008→T009 are mandatory red/green pairs.

## Phase 3: US4 — Generate trustworthy missing evidence offline (Priority: P1)

**Goal**: Deliver the separate offline producer prerequisite without extending the god module.

**Independent test**: In a temporary workspace, generate a compatible pack, reload it, inject a failure before publication, and rerun unchanged with fit counters. Original source/artifacts remain unchanged and repeated valid generation has zero fits.

- [X] T010 [US4] Add failing paired-fold evaluation and frozen candidate test-diagnostic tests in `tests/test_ui_evidence_training.py`; assert shared exact membership, train/validation scores from one fitted model, DEV-only preprocessing, actual parameters/dimensions, `ddof=0` aggregation and no test-target-driven configuration choices.
- [X] T011 [US4] Implement bounded paired train/validation evaluation and RF per-fold encoded/raw-family importance capture in new `src/ai_job_market/ui_evidence_training.py`, reusing existing public pipeline/split/metric helpers unchanged; pass fold/membership portions of T010 without monkeypatching or editing `core.py`.
- [X] T012 [US4] Add frozen five-candidate historical test evaluation and pre-test declaration production in `src/ai_job_market/ui_evidence_training.py`; pass T010, keep retrospective metrics separate from CV ranking, and never replace the saved selected full model.
- [X] T013 [US4] Extend `tests/test_ui_evidence_training.py` with failing full/Top-2 frozen-setting, two-input bundle, q90, permutation/encoded importance, deterministic benchmark and reload tests; prove test-target/error changes do not change configuration, policy or selected source-row offsets (snapshot IDs may change) and mismatched full evidence is rejected.
- [X] T014 [US4] Implement full/Top-2 CV, Top-2 all-DEV fit/serialization data, matched-population historical predictions/bands/importance and target-independent benchmark selection in `src/ai_job_market/ui_evidence_training.py`; pass T013 while loading/verifying—not refitting—the existing full bundle and preserving all historical test rows.
- [X] T015 [US4] Add failing CLI tests in `tests/test_ui_evidence_cli.py` for workspace preflight, read-only `--check`, per-component reuse, no-fit unchanged rerun, schema/policy-only rebuild, failed generation and atomic current-pointer preservation; use temporary workspaces and counting estimators.
- [X] T016 [US4] Implement the thin new `src/ai_job_market/ui_evidence.py` command, source/prepared-data reconciliation, component orchestration, staging/publication and explicit statuses/exit codes; pass T015 and prohibit implicit main-pipeline runs, arbitrary pickle paths or overwriting legacy packs.
- [X] T017 [US4] Add failing inherited-tuning snapshot/configuration conflict cases to `tests/test_ui_evidence_contract.py`, then implement validated optional history import and generated HTML references in `src/ai_job_market/ui_evidence.py`; preserve unavailable reasons and label inherited non-nested/R² methodology rather than rerunning tuning or copying unsupported narrative.
- [X] T018 [US4] Run isolated end-to-end generation/reload/failure/reuse tests in `tests/test_ui_evidence_cli.py`, reconcile fixture metrics against independent calculations and record results plus unchanged protected-file hashes in `specs/003-improve-model-ui/verification.md`; do not generate the real pack yet.

**Checkpoint**: US4 works independently in fixtures. Five-candidate and variant evaluations have honest identities; models/policies are reusable and no historical test row is labelled pristine. Core/main pipeline remain unchanged.

## Phase 4: US1 — Compare model families (Priority: P1, first UI MVP)

**Goal**: Four-tab chart-first Page 04, with evidence-qualified interpretation rather than static winner/fit claims.

**Independent test**: Open Page 04 with fixture packs whose ranks, model counts, fold counts and missing fields vary; every figure/table must reconcile and required unavailable states must render without fitting.

- [X] T019 [P] [US1] Extend `tests/test_model_evidence_views.py` with failing train/validation combo, ratio/R² axes, Dummy-reference, ranking/tie, fold timeline, RF drift/family aggregation and GB head-to-head/MedAE tests; cover missing/zero train scores and a non-RF first rank.
- [X] T020 [P] [US1] Add failing Page 04 AppTest/entrypoint navigation cases in `tests/test_model_ui_pages.py`, with five chips/four tabs, chart-before-table order, local missing-family states and fit/tune/producer calls forbidden; use test-only controlled auth/workspace fixtures.
- [X] T021 [US1] Extend `src/pages/model_evidence.py` with Page 04 views and deterministic baseline-status rules from `contracts/scenario-ui.md`; pass T019 without hardcoded counts, parameters, mean importances, winner labels or causal fit assertions.
- [X] T022 [US1] Reorganize `src/pages/page04_model_comparison.py` into the ribbon and four required tabs, charts first and collapsed evidence downloads after them; pass T020 and preserve the existing renderer/router interface.
- [X] T023 [US1] Verify the Page 04 MVP independently with focused tests and document its chart/source/configuration mappings and review caveats in `docs/MODEL_UI.md`; record the check in `specs/003-improve-model-ui/verification.md` before beginning the next UI slice.

**Checkpoint — MVP**: US4 + US1 deliver a standalone truthful comparison page. It is valid to review this slice before further UI work, without retraining the primary pipeline.

## Phase 5: US3 — Controlled salary scenarios (Priority: P1)

**Goal**: Safe two-input serving, common strict validation, explicit labelled experience exceptions and trustworthy exports.

**Independent test**: On a compatible fixture pack, add/quick-load/run/clear scenarios and export matching results; missing/incompatible Top-2 must disable inference. Workspace or queue mutations cannot show stale results.

- [X] T024 [US3] Add failing inference/export tests in `tests/test_salary_inference.py` for exact feature ordering, batch revalidation, finite shape/output, bounds/clipping, known/unknown/zero actuals, signed variance, source `AI Engineering` mapping and ordered v1 CSV schemas; assert invalid inputs cause zero predict calls and all cases zero fit calls.
- [X] T025 [US3] Implement batch prediction, immutable benchmark actual lookup, result math and pure audit/source-schema CSV builders in new `src/ai_job_market/salary_inference.py`; pass T024 without full-bundle fallback, fabricated source attributes, repurposed `error_pct` or artifact writes.
- [X] T026 [US3] Add failing strict/default and acknowledged experience-only growth tests in `tests/test_salary_inference.py`; verify validated grid/endpoints, separate requested prediction calls, overlapping category curves, exception labels and invariant category/pair checks.
- [X] T027 [US3] Implement validated on-request growth inputs/results in `src/ai_job_market/salary_inference.py`; pass T026 while keeping default curves within title support and optional 0–15 extensions explicitly flagged as uncalibrated extrapolation.
- [X] T028 [US3] Extend `tests/test_model_ui_pages.py` with failing Page 06 tests for reactive dependent widgets, fractional medians, quick-load identity, duplicate IDs, 0/100/101 queues, strict/exception acknowledgements, model-unavailable states, session isolation and queue/workspace/evidence invalidation.
- [X] T029 [US3] Implement reactive builder, benchmark strip, common policy badges, session-scoped queue/action lifecycle and explicit exception controls in `src/pages/page06_prediction.py`; pass interaction/state portions of T028 with no free-text scenario controls and no automatic inference on unrelated reruns.
- [X] T030 [US3] Add failing prediction/whisker/actual-marker/paging assertions to `tests/test_model_evidence_views.py`, then implement result/growth figure builders in `src/pages/model_evidence.py` and chart-first results/download wiring in `src/pages/page06_prediction.py`; all 100 rows remain in table/export while the chart pages ten at a time.
- [X] T031 [US3] Run the Page 06 policy/inference/AppTest matrix and reconcile charts, results and download bytes; document strict defaults, labelled exceptions, new signed audit CSV, original-schema limitations and invalidation rules in `docs/MODEL_UI.md` and record evidence in `specs/003-improve-model-ui/verification.md`.

**Checkpoint**: US3 serves only a verified Top-2 contract. Ordinary builder, benchmark and curve flows share bounds; exceptions are acknowledged and cannot bypass missing models or unobserved pairs. Page 05 is not a runtime prerequisite.

## Phase 6: US2 — Selected-model diagnostics (Priority: P2)

**Goal**: Five sections distinguish final configuration, historical test performance, inherited tuning, feature reliance and uncertainty.

**Independent test**: Render Page 05 from matching full/Top-2 fixtures and optional missing/conflicting tuning history; numeric deltas, populations and applied settings must remain correct without affecting inference.

- [X] T032 [P] [US2] Extend `tests/test_model_evidence_views.py` with failing final-config CV/test matching, dollar/R² separate scales, signed/relative delta, residual/q90/coverage, four sensitivity charts, raw-vs-encoded importance and matched-population variant comparison cases.
- [X] T033 [P] [US2] Extend `tests/test_model_ui_pages.py` with failing Page 05 cases for six chips/five sections, collapsed methodology, chart-first detail, applied-vs-sweep parameter distinction, historical example labels and local unavailable tuning/importance states.
- [X] T034 [US2] Implement Page 05 scoreboard, tuning, variant/importance and trust view builders in `src/pages/model_evidence.py`; pass T032 and ensure baseline family CV cannot silently replace the final configuration's CV metrics.
- [X] T035 [US2] Reorganize generalization, actual/predicted/residual and tuning sections in `src/pages/page05_best_model.py`; pass corresponding T033 cases, removing hardcoded 295-row/parameter/proven-no-overfit narratives while preserving actual evidence downloads.
- [X] T036 [US2] Implement feature reliance, fixed Top-2/full comparison and uncertainty/historical-audit sections in `src/pages/page05_best_model.py`; complete T033 with matching model-specific q90, raw/encoded labels and honest inclusion/exposure flags rather than pristine/production claims.
- [X] T037 [US2] Run focused Page 05 tests and document historical population, tuning limitations, chart sources and q90 interpretation in `docs/MODEL_UI.md`; record metric-to-source reconciliation in `specs/003-improve-model-ui/verification.md`.

**Checkpoint**: All four stories are independently testable. No remaining page uses stakeholder literals as model evidence or calls an offline producer during interaction.

## Phase 7: Release artifacts and cross-cutting verification

**Goal**: Generate necessary real supplemental evidence once, validate consumers, review the actual browser UI and synchronize documentation.

- [X] T038 Add read/predict-only release assertions in new `tests/test_ui_evidence_release.py`, opt-in through `UI_EVIDENCE_WORKSPACE`, checking model/policy/metric identities, equal full/Top-2 test rows, historical example inclusion, reload parity and protected baseline hashes; absent required evidence must fail this opted-in check without generating it.
- [X] T039 Update only obsolete widget-count/mandatory-prose assertions for Pages 04–06 in `tests/test_release_contract.py` to behavior/contract checks supplied by `tests/test_model_ui_pages.py`; keep unchanged-page expectations, leakage gates and baseline artifact coverage intact.
- [X] T040 After producer tests pass, use new `src/ai_job_market/ui_evidence.py` to generate only missing/invalidated real components under `outputs/ui_evidence/` and `artifacts/ui_evidence/`; record current evidence ID, actual metrics, inherited/unavailable sections and baseline hash parity in `specs/003-improve-model-ui/verification.md` without running root `pipeline.py`.
- [X] T041 Run CLI `--check`, the opted-in `tests/test_ui_evidence_release.py`, and an unchanged normal invocation; prove reuse/no-fit through `tests/test_ui_evidence_cli.py` spies and record actual saved-pack identity/prediction equivalence in `specs/003-improve-model-ui/verification.md`.
- [X] T042 Run the focused/new suites, relevant existing tests, full pytest and targeted Ruff/diff checks from `specs/003-improve-model-ui/quickstart.md`; record counts, skips and any inherited failures in `specs/003-improve-model-ui/verification.md` without weakening unrelated tests or regenerating valid packs.
- [X] T043 Exercise actual `streamlit.py` in a real browser at 1280×800 and 1440×900, including all three pages, roles, keyboard labels, 1/3/100 scenarios, paging, missing packs, exception toggles and CSV downloads; save screenshots/check results in `specs/003-improve-model-ui/verification.md`, or explicitly leave visual acceptance pending if tooling is unavailable.
- [ ] T044 Measure cold load, warm three-row inference/visible rendering and requested growth separately, and conduct the two 60-second reviewer walkthroughs from `specs/003-improve-model-ui/quickstart.md`; record measured outcomes/limitations in `specs/003-improve-model-ui/verification.md`, not an assumed sub-0.01-second promise.
- [X] T045 Finalize `docs/MODEL_UI.md` and README links with the real offline command, reuse/invalidation rules, generated chart paths, source-to-metric map, CSV migration and scientific limitations; synchronize `specs/003-improve-model-ui/spec.md`, `plan.md` and contracts if verified implementation details materially differ, obtaining approval before scope changes.
- [X] T046 Run `graphify update .` after verified code changes (or documented equivalent/valid exception) and record affected relationships and command outcome in `specs/003-improve-model-ui/verification.md`; do not retrain models to refresh the graph.
- [X] T047 Review the actual source/spec diff, recheck protected hashes and producer/consumer fingerprints, confirm zero new logic in `src/ai_job_market/core.py`, and reconcile all task statuses against evidence in `specs/003-improve-model-ui/tasks.md` and `verification.md`; report unresolved browser/governance gates before requesting merge approval.

## Phase 8: Visual-priority and semantic-color amendment

**Goal**: Match the stakeholder document's visual-first intent more closely without restoring unsupported scientific claims: stable semantic colors, red historical-test error columns, and chart width proportional to decision priority.

**Independent test**: Render Pages 04–06 from reordered fixture traces and the real evidence pack. Semantic role colors must remain stable, all roles retain non-color encoding, and each section has at most one full-content-width P1 chart at 1280×800 and 1440×900.

- [X] T048 Add failing semantic-role color and chart-priority tests in `tests/test_model_evidence_views.py`, covering Page 04 validation/train/Dummy, Page 05 historical-test/DEV-CV, and Page 06 prediction/uncertainty/actual traces; reorder inputs to prove colors are not positional.
- [X] T049 Add the shared evidence-role palette and minimal figure/layout metadata helpers to `src/pages/model_evidence.py`; pass T048 without CSS, a new dependency, or changes to evidence calculations.
- [X] T050 Update `src/pages/page04_model_comparison.py` to apply the contracted colors and P1/P2/P3 sizing: full-width primary overall combo, bounded ratio/R², and compact timeline/drift/supporting views; preserve chart-first ordering and all evidence tables/downloads.
- [X] T051 Update `src/pages/page05_best_model.py` so historical-test dollar errors are red columns, DEV-CV is an orange comparison line, R² is compact, tuning sensitivities use a readable two-column grid, and residual/ECDF/encoded-importance charts no longer consume full width unless label tests require promotion.
- [X] T052 Update `src/pages/page06_prediction.py` and the shared prediction figure so prediction bars are blue/teal, empirical uncertainty is amber, actuals are red diamond markers, and requested growth is a P2 bounded view; preserve queue/inference/export behavior.
- [X] T053 Extend `tests/test_model_ui_pages.py` with AppTest assertions for chart ordering, trace roles and native container structure; inspect the real app at 1280×800 and 1440×900 when browser tooling is available, recording any justified priority promotion or leaving pixel acceptance explicitly pending.
- [X] T054 Update `docs/MODEL_UI.md` with the semantic palette and priority-width map, run focused tests, full pytest, Ruff and `ui_evidence --check`, and confirm no model evidence regeneration occurred unless a producing fingerprint actually changed.
- [X] T055 Re-run five-axis review, protected-hash checks and `graphify update .`; record visual amendment results in `specs/003-improve-model-ui/verification.md` and reconcile all task statuses. T044 remains a separate independent-human acceptance gate.

**Checkpoint outcome**: The stakeholder explicitly approved T048–T055 on 2026-09-19. The amendment is implemented and verified; pixel-level browser acceptance remains documented as pending.

## Phase 9: Report readability and third-party comprehension amendment

**Goal**: Eliminate clipped/overlapping report values and add enough evidence-derived explanation for a third party to understand Pages 04–06 without source/CSV knowledge.

**Research basis**: `ui-readability-audit.md` and R13–R16 in `research.md`. Runtime inspection found one forced 20/20/55/25 margin profile for all charts, excessive direct-label density in compact views, inconsistent table formatting, and missing finding/meaning/limit conclusions.

**Independent test**: At 1280×800 and 1440×900, a fresh reviewer identifies each page's main finding, two values and key limitation within 90 seconds. Browser bounding boxes show no required label collision/clipping. Automated tests reconcile conclusions and label policies to fixture evidence.

- [X] T056 Add failing figure-readability tests in `tests/test_model_evidence_views.py` and a focused shared-render test file if needed, covering per-figure height/margins, categorical automargin, non-scientific USD ticks, long-label handling, dense scatter text suppression and maximum direct-label counts.
- [X] T057 Implement minimal readability-profile helpers in `src/pages/model_evidence.py` and change `src/pages/common.py::show_plot` to preserve explicit figure geometry while merging safe defaults; pass T056 and regression-test untouched figure behavior without CSS or a new chart dependency.
- [X] T058 Add failing pure conclusion tests for Page 04 ranking/Dummy/fold findings, Page 05 CV-test/q90/variant/tuning findings and Page 06 batch/actual/exception findings; include changed-rank, missing evidence, zero reference and nonfinite cases.
- [X] T059 Implement small deterministic conclusion builders and a native Streamlit conclusion renderer in `src/pages/model_evidence.py` or one focused presentation module; pass T058 with `Finding`, `Why it matters`, `Limit`, optional `Decision/use` and source identity, never hardcoded scientific outcomes.
- [X] T060 Update `src/pages/page04_model_comparison.py` with a plain-language page brief/glossary, section questions, derived ranking/baseline/fold conclusions, reduced drift/fold label density and reader-formatted supporting tables; preserve evidence downloads and scientific caveats.
- [X] T061 Update `src/pages/page05_best_model.py` with a page brief/glossary, CV-test/uncertainty/variant/tuning conclusions, readable scorecard/audit tables and tuning layout that stacks when half-width cannot meet the profile; preserve historical-exposure and non-nested limitations.
- [X] T062 Update `src/pages/page06_prediction.py` with an active-model/domain brief and a post-run snapshot-bound conclusion covering scenario count/range, q90 basis, known actual availability and exception count; use short chart IDs and full profile hover/table detail without changing inference/export behavior.
- [X] T063 Extend `tests/test_model_ui_pages.py` with real-entrypoint assertions for brief → chart → conclusion → table ordering, glossary availability, changed-evidence conclusions, unavailable states and no stale Page 06 conclusion after queue/context mutation.
- [X] T064 Add table/readability contract assertions for reader-facing column names, units/precision, Top-N disclosure, pinned identity columns where supported and full-row download preservation; fix only Pages 04–06 tables required by the amended spec.
- [ ] T065 Run Chrome DevTools browser review at 1280×800 and 1440×900 across every Page 04–06 tab plus 1/3/10-scenario Page 06 results; record screenshots, label bounding boxes, console/accessibility findings and third-party walkthrough results. If DevTools MCP remains unavailable, leave T065 and pixel acceptance open rather than claim success from AppTest.
- [X] T066 Update `docs/MODEL_UI.md`, the UI audit and verification record with the implemented reading path, conclusion source map and browser evidence; run focused/full tests, Ruff and `ui_evidence --check`. If shared generated-chart fingerprints change, regenerate one versioned supplemental pack and prove the next invocation reuses it.
- [X] T067 Complete five-axis review, protected baseline hashes, Graphify refresh and task/spec reconciliation; report T044/T065 or any other unresolved human/browser gate explicitly.

**Checkpoint outcome**: The stakeholder explicitly approved T056–T067. Automated/source work is complete; T065 remains open because Chrome DevTools MCP is unavailable and AppTest cannot prove pixels/accessibility.

## Phase 10: Consistent tables and training-method audit amendment

**Goal**: Make Page 04–05 tables consistently emphasize governing versus contextual evidence and add detailed, collapsible, exportable explanations of the actual temporal-validation and tuning procedures.

**Research basis**: R17–R20 in `research.md`, FR-037–FR-043, and the existing training-audit contract in `docs/TRAINING_AUDIT.md`. The installed Streamlit API confirms that `st.table` renders Markdown for small static tables while `st.dataframe` does not render inline Markdown.

**Independent test**: With fixture and real evidence, Page 04–05 compact tables apply the same documented bold/italic grammar, detailed tables keep consistent units/precision, temporal/tuning guides describe the actual implementation, and downloads byte-match current evidence or a compatible complete audit run. Missing/incompatible logs stay optional and cause zero training calls.

- [X] T068 [US7] Record explicit approval for T068–T079 and refresh protected-file/evidence/log hashes in `specs/003-improve-model-ui/verification.md`; confirm active pack `ui-b35971c9f8bcd166` remains valid and no producer/data/config/lock contract changed before source edits.
- [X] T069 [US7] Add failing pure presentation tests in `tests/test_model_evidence_views.py` for compact decision-table rows, Markdown-safe emphasis, tie handling, Page 04 minimum-unrounded-MAE winners, Page 05 maximum-unrounded-R² stage winners, applied-vs-sweep distinction and consistent USD/R²/percent/duration/count formatting.
- [X] T070 [US7] Implement the smallest shared table builders/formatters in new presentation-only `src/pages/model_training_presentation.py`; pass T069 using `st.table`-ready display frames for small summaries and plain typed frames/config metadata for detailed `st.dataframe` views, with no CSS or scientific-value mutation and without invalidating the evidence-producing `model_evidence.py` fingerprint.
- [X] T071 [US7] Add failing methodology and audit-loader tests in `tests/test_model_evidence_views.py` and a focused new `tests/test_training_audit_ui.py` if separation is clearer; cover actual adjacent-block fold facts, anchor/sweep counts, ranking metric/direction, applied-selection provenance, unsupported rationale exclusion, fixed log-root containment, symlink/escape refusal, source mismatch, incomplete manifest and bad checksum.
- [X] T072 [US7] Implement pure temporal/tuning guide-data builders and a read-only compatible audit-trace loader in `src/pages/model_training_presentation.py`; pass T071, keep the active supplemental pack authoritative, expose run IDs/status/download descriptors only, and invoke no fit/tune/producer path.
- [X] T073 [US7] Extend `tests/test_model_ui_pages.py` with failing Page 04 AppTest cases for guide placement before tabs, collapsed detailed `What/How/Why/Limits` content, correct sliding-window wording, table legend/emphasis, detailed-table precision and fold metrics/membership plus optional audit downloads.
- [X] T074 [US7] Update `src/pages/page04_model_comparison.py` to render the high-priority temporal-validation expander and two-tier consistent tables; pass T073, preserve chart/conclusion ordering and existing caveats, and label audit logs as a distinct historical primary-pipeline trace.
- [X] T075 [US7] Extend `tests/test_model_ui_pages.py` with failing Page 05 AppTest cases for anchor/coordinate tuning explanation, stage trial/search-space facts, CV R² versus family-MAE distinction, applied-selection source, non-nested/test-use limits, shared table grammar and complete tuning/audit download availability.
- [X] T076 [US7] Update `src/pages/page05_best_model.py` to render the expanded tuning guide and consistent compact/detail tables; pass T075, exclude legacy `tuning_rationale`, distinguish stage winners from applied settings and preserve all generalization/exposure limitations.
- [X] T077 [US7] Add byte-for-byte download and no-training assertions across `tests/test_model_ui_pages.py`, `tests/test_training_audit_ui.py` and `tests/test_release_contract.py`; prove current CSVs and validated audit files are exported unchanged, missing/incompatible logs remain local unavailable states, and ordinary page renders call no fit/tune/producer function.
- [X] T078 [US7] Update `docs/MODEL_UI.md`, `specs/003-improve-model-ui/ui-readability-audit.md` and `specs/003-improve-model-ui/verification.md` with table semantics, exact fold/tuning reading guides, source distinctions and export map; run focused/full pytest, Ruff, `git diff --check` and `ui_evidence --check`, reusing the current pack unless a producing fingerprint actually changes.
- [ ] T079 [US7] Review the final source/spec diff, recheck protected hashes, run `graphify update .`, and attempt browser/third-party acceptance for the new tables/expanders at 1280×800 and 1440×900; if Chrome DevTools or independent reviewers remain unavailable, keep T044/T065 and the corresponding portion of T079 open rather than inferring acceptance from AppTest.

**Implementation outcome**: The user explicitly approved T068–T079 with conditional training only if evidence was missing. Existing evidence/logs were sufficient, so T068–T078 were completed without training or regeneration. T079 browser/human acceptance remains open because those tools/reviewers are unavailable.

## Dependencies & Execution Order

```text
User implementation approval
  → Setup T001–T003
  → Foundations T004–T009
  → US4 producer T010–T018
  → US1 comparison MVP T019–T023
  → US3 prediction T024–T031
  → US2 diagnostics T032–T037
  → Real generation + release verification T038–T047
  → Stakeholder visual amendment approval
  → Semantic colors and priority sizing T048–T055
  → Stakeholder readability amendment approval
  → Readability and derived conclusions T056–T067
  → Stakeholder table/training-method amendment approval
  → Consistent tables and audit guides T068–T079
```

- No task runs merely because this task list exists. T001, T048 and T056 had separate approvals; T068 is separately gated by explicit approval of the table/training-method amendment.
- US4 is the shared evidence prerequisite. UI stories may use fixtures independently, but real release claims require T040–T041.
- US1/US3 are both P1; US1 is first for a smaller UI MVP. US2 is P2. Story numbering follows the spec, so execution deliberately uses US4 → US1 → US3 → US2.
- Tests must fail for the missing behavior before the corresponding implementation. A skipped release check is not a pass; opted-in real checks fail if the pack is absent.
- T012 depends on T011; T014 on T013; T016 on T015 and generation/policy/io foundations; T017 on T016; T018 closes the producer checkpoint.
- T021 follows T019; T022 follows T020–T021. T025 follows T024, T027 follows T026, T029 follows T028 and serving helpers; T030 completes the results path before T031.
- T034 follows T032, T035/T036 follow T033–T034. T040 requires completed producer and figure-generation paths plus T038; T041 requires T040. Browser/performance checks reuse T040 artifacts.
- Shared `src/pages/model_evidence.py` and `tests/test_model_ui_pages.py` edits are serialized across stories. Do not run all UI implementations in parallel merely because the pages differ.
- T069→T070 and T071→T072 are mandatory red/green pairs. T073→T074 and T075→T076 follow the shared helpers. T077 validates exports/no-training before documentation/release review in T078–T079.

## Parallel Examples

`[P]` means distinct files and no dependency on the paired incomplete task, not permission to ignore phase prerequisites.

- **US4**: Keep fitting/orchestration work sequential; a reviewer can inspect the evidence contract while T010/T013 tests are authored, but no producer tasks are marked parallel because they share model/state assumptions.
- **US1**: After T018, T019 (`tests/test_model_evidence_views.py`) and T020 (`tests/test_model_ui_pages.py`) can run concurrently; both must finish before their implementations.
- **US3**: Keep T024–T030 sequential because inference, policies and page state are coupled. Once serving contracts pass, a separate reviewer can check CSV examples in `docs/MODEL_UI.md` without editing shared implementation files.
- **US2**: After earlier slices are verified, T032 (`tests/test_model_evidence_views.py`) and T033 (`tests/test_model_ui_pages.py`) can run concurrently; coordinate subsequent changes to the shared view module.

## Traceability

| Requirement group | Tasks / verification |
| --- | --- |
| FR-001/015/025/026; US4; SC-009 | T001–T018, T038, T040–T042, T047 |
| FR-002–004; evidence/visual integrity; SC-001/002/006 | T004–T005, T008–T009, T019–T023, T030, T032–T037, T043 |
| FR-005–008; US1 | T019–T023 |
| FR-009–013; US2 | T032–T037, T043–T044 |
| FR-014/018–023; US3; SC-004/005/007 | T024–T031, T038, T041–T044 |
| FR-016/017; SC-010 | T006–T007, T013–T014, T026–T031, T043 |
| FR-024 and documentation/governance | T002, T023, T031, T037, T045–T047 |
| SC-003 reviewer time | T043–T044 |
| SC-008 no new UI training / preserved artifacts | T010–T018, T020, T024, T028, T033, T038–T042, T047, T068, T072, T077–T079 |
| FR-037–FR-043; US7; SC-017–SC-020 | T068–T079 |

## Implementation Strategy and Review Gate

1. Deliver contracts/policy and a fixture-verified offline producer first; this makes the highest-risk provenance/feature mismatch testable before UI polish.
2. Deliver Page 04 as the first UI MVP, then Page 06 serving, then Page 05 historical diagnostics. Verify each slice before continuing.
3. Generate the real new pack only after producer/consumer tests pass. Reuse it for all subsequent review/demo work unless a producing dependency/contract changes; regenerate or invalidate only affected components.
4. Preserve inherited scientific limitations and the unrelated global upload-training issue as documented risks. Do not repair them silently, manufacture pristine data or choose new model settings to reproduce stakeholder example numbers.
5. Mark tasks `[X]` only with evidence, never on planning completion. Keep living artifacts synchronized.

**Implementation outcome to date**: Explicit user approval was recorded before T001, T048, T056 and T068. Baseline, visual-priority, readability and table/training-method automated verification are complete. T044, T065 and the browser/human portion of T079 remain open acceptance gates. See `verification.md`.
