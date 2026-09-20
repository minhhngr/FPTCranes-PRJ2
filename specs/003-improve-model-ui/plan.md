# Implementation Plan: Evidence-Backed Model UI and Offline Supplemental Artifacts

**Branch**: `005-improve-model-ui` | **Date**: 2026-09-19 | **Spec**: [spec.md](spec.md)  
**Input**: `specs/003-improve-model-ui/spec.md`  
**Status**: Baseline, semantic-color, readability/conclusion and approved table/training-explanation amendments implemented; automated verification complete, browser/human acceptance pending.

## Summary

Redesign Pages 04–06 as chart-first, evidence-driven views and add a bounded offline command to fill missing/unverifiable evidence. The user approved this scope expansion on 2026-09-19 with a strict architectural constraint: **new generation logic belongs in new `.py` files; do not edit `src/ai_job_market/core.py`.** Preserve the existing main pipeline, full-feature bundle, data split, tuning history and unrelated pages.

The supplemental workflow reuses existing public training helpers but owns its paired train/validation evaluator, fixed Top-2 experiment, historical benchmark examples, provenance and publication. UI consumers validate immutable packs and run only prediction. Default scenario/benchmark/curve bounds are uniform and DEV-derived; experience-only exceptions are opt-in, visibly labelled and cannot bypass categorical or artifact checks. Title/category relationships are observed pairs, not curated business logic.

The source document's exact numbers are not acceptance targets. Existing test data was already scored: three generated examples remain historical benchmark records, **not pristine reserved rows**, and the test denominator remains the actual full partition (currently 298). [research.md](research.md) records decisions and rejected alternatives; [evidence-review.md](evidence-review.md) records source discrepancies.

## Technical Context

**Language/Version**: Python 3.13.12 in `.venv`; Ruff target py312.  
**Primary Dependencies**: Existing Streamlit 1.64.0, Plotly 7.1.0, pandas 2.3.3, NumPy 2.5.3, scikit-learn 1.9.1, joblib 1.6.0, PyYAML; standard-library argparse/hashlib/json/pathlib/dataclasses. No dependency additions/upgrades.  
**Storage**: Existing CSV/JSON/joblib inputs preserved; immutable supplemental CSV/JSON/HTML under `outputs/ui_evidence/<evidence_id>/`, generated models under `artifacts/ui_evidence/<evidence_id>/`, atomic `outputs/ui_evidence/current.json`.  
**Testing**: pytest 9.1.1, estimator spies and temporary-workspace producer tests, Streamlit AppTest via the actual entrypoint, pure figure/CSV assertions, real-browser visual checks.  
**Target Platform**: Existing Linux offline CLI and desktop web UI.  
**Project Type**: Offline ML reporting application with saved-artifact consumers.  
**Performance Goals**: Target warm three-scenario visible results within one second; measure cold-load/inference/render separately. Zero fitting for ordinary page actions and for unchanged producer reruns. Max queue 100, charts show 10 scenarios per page by default without truncating exports.  
**Constraints**: No `core.py`, primary pipeline, source data, dependency lock, model-selection or test-partition changes. No fresh tuning, test-driven model promotion, manufactured pristine data, auth/navigation redesign or new global upload processing path.  
**Scale/Scope**: Three pages; five candidate families; current five chronological DEV folds; two final feature variants; up to three historical examples; current 1,201 DEV/298 test rows. All counts derive from inputs. Initial supplemental fits are bounded: five candidates × folds, frozen full and Top-2 × folds, five candidate all-DEV fits, one Top-2 all-DEV fit (41 at five folds); reuse equivalent component results where identity is exact. Never refit the saved full bundle just to serve it.

## Constitution Check

**Pre-research gate**: New-work design passes leakage, artifact separation, offline fitting and input-validation requirements. Known historical VI.3/VI.4 and global upload boundary discrepancies are explicitly bounded below; no unconditional legacy compliance claim is made.  
**Post-design gate**: Same disposition. New evaluations use fixed configurations, DEV-only fitting and a pre-test declaration, not a new search/selection procedure. Candidate test diagnostics add reporting coverage without rewriting old artifacts or claiming unseen-test purity. Approval of tasks is required before execution; inherited exceptions must remain visible in review.

| Gate | Design and acceptance evidence | State |
| --- | --- | --- |
| Data integrity | Target `annual_salary_usd`; original 13-feature and declared two-feature policies; original DEV/test row membership; fold-local preprocessing; DEV-only role policy; test targets never used for configuration/example selection | Design pass; leakage spies and mutation tests planned |
| Reproducibility | Seed from existing config; exact params and feature order; source/data/lock/producer/contract hashes; immutable membership/metrics, dependency-aware reuse | Design pass; real output generation pending |
| Artifact contracts | New namespace and versioned schemas; baseline pack/bundle hashes unchanged; missing/unverifiable supplemental evidence regenerated or explicitly unavailable; no silent fallback | Design pass; producer-consumer checks planned |
| Evidence-first | Test failures before each changed behavior; fixture/estimator tests, real saved-artifact checks, UI and browser evidence | Planned, not executed |
| Streamlit boundary | New pages and helpers import only read/validate/predict paths; no fit/tuning/generation action; existing global upload control outside scope | New paths pass by design; inherited global issue disclosed |
| Universal validation | CLI paths and flags, artifact paths/IDs/hashes, filter enums, observed pairs, finite years, queue length, immutable benchmark lookup, scoped exception acknowledgements, session revisions | Design pass; negative tests planned |
| Candidate ladder | Same splits for five frozen factory candidates; CV and historical test MAE/RMSE/R²/MedAE; display ranking independent of unchanged saved selection | Supplemental coverage planned |
| Tuning and selection | Existing two-stage DEV/R² history displayed as inherited, non-nested; no new tuning; fixed Top-2 is not claimed optimal or test-selected; no automatic primary model replacement | Bounded inherited VI.3/VI.4 exception below |
| Importance/uncertainty | Both encoded and raw permutation evidence for full/Top-2; RF fold drift; empirical test q90 labelled with same-population coverage caveat | Design pass; no causal/production claim |
| Documentation/visualization | Source-to-chart map, generated HTML figure references in supplemental outputs, CLI/exports/domain guide and verification record | Planned |
| Graphify/Karpathy | Existing graph queried twice, direct source follows truncation; focused new modules and preserved `core.py` | Planning review complete; refresh after code changes |

## Project Structure

### Documentation (this feature)

```text
specs/003-improve-model-ui/
├── spec.md
├── evidence-review.md
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   ├── evidence-pack.md
│   └── scenario-ui.md
├── checklists/requirements.md
└── tasks.md
.specify/assessments/improve-model-ui/research.md
AGENTS.md                              # update marked plan link only
```

### Source Code (repository root)

```text
src/ai_job_market/
├── core.py                            # UNCHANGED; import established helpers only
├── ui_evidence.py                     # NEW thin CLI/orchestration/publication
├── ui_evidence_training.py            # NEW bounded diagnostic evaluation + Top-2 fits
├── ui_evidence_io.py                  # NEW schemas, identities, validation, read/write helpers
├── scenario_policy.py                # NEW DEV observed pairs/bounds + shared validation
└── salary_inference.py               # NEW prediction, curve and export helpers; no fitting
src/pages/
├── model_evidence.py                 # NEW cached loaders + pure evidence/figure builders
├── model_training_presentation.py    # NEW table/method/audit-log presentation readers
├── page04_model_comparison.py         # four tabs, evidence-based labels
├── page05_best_model.py               # five sections, historical diagnostics
├── page06_prediction.py               # reactive builder, queue and safe Top-2 serving
└── common.py                         # only modernize touched shared plot width argument
streamlit.py                          # preserve routing/render signatures
pipeline.py                           # UNCHANGED primary training command
config/project.yaml                   # UNCHANGED
requirements.txt, uv.lock             # UNCHANGED; preserve inherited untracked lock

tests/
├── test_ui_evidence_contract.py       # NEW manifest/path/provenance/publication/reuse
├── test_ui_evidence_training.py       # NEW folds, fit scopes, paired scores, fixed settings
├── test_ui_evidence_cli.py            # NEW isolated CLI, partial failure, no-op rerun
├── test_scenario_policy.py            # NEW observed pairs/bounds/exception matrix
├── test_salary_inference.py           # NEW no-fit serving, curves, batch/CSV math
├── test_model_evidence_views.py       # NEW figure/table sources and diagnostics
├── test_model_ui_pages.py             # NEW AppTest navigation/pages/session cases
├── test_ui_evidence_release.py        # NEW opt-in real-pack read/predict checks, never fit
└── test_release_contract.py           # update only obsolete presentation assertions

docs/MODEL_UI.md                       # NEW CLI, UI, evidence/CSV/domain guide
README.md                             # entrypoint/guide link; no numeric promises
```

**Structure Decision**: Five focused new backend Python modules plus one small UI support module. No plugin system, service/API, database, background queue, estimator wrapper, global trainer extraction or monolithic replacement. Keep page `render(st, root, role)` signatures because the existing router consumes them. The Streamlit skill's preference for page scripts does not justify breaking this established interface.

## Phase 0 — Research and scope resolution

Q1–Q3 are resolved in the spec and evidence review. Research was read-only, with no model fitting or inference. Additional findings affecting implementation:

1. Current factory RF/GB parameters differ from the source document; regenerated candidate numbers may change. The UI must distinguish 'lowest supplemental CV MAE' from 'saved selected full model'.
2. `core.make_model_pipeline` already supports Top-2 feature subsets. New paired-score evaluation belongs in `ui_evidence_training.py`; no monkeypatch or edit to `core.evaluate_model_cv`.
3. Historical 298-row test exposure cannot be undone. Examples retain historical scoring inclusion and use target-independent row selection.
4. DEV observed pairs are inconsistent, and medians may be fractional. Use DEV-only observed policy with honest labels, not a hand-built business mapping or integer truncation.
5. `SOURCE_COLUMNS` uses `AI Engineering` for category. Exports must map back explicitly and preserve original order.
6. Existing source-text tests count widgets/interpretation calls. Replace only affected brittle assertions with behavior checks while preserving leakage/artifact assertions and untouched-page requirements.

## Phase 1 — Design

### Offline data boundary and component lifecycle

CLI accepts a local workspace root; source-code/config/lock identities come from the checkout, input artifact paths from that workspace. Reject missing/escaping/symlinked artifact destinations and malformed stage/run selectors before writes. Input paths are fixed relative to the workspace, not arbitrary user-specified CSV/model paths. Existing prepared DEV/test data must match source preparation and temporal summary; read-only cleaning/split comparison may reuse existing functions without fitting. If source or prepared evidence cannot be reconciled, block generation and report the mismatch rather than rerun the primary pipeline automatically.

Compute component dependencies before work. Matching immutable components are reused after schema/hash validation. For changed/missing components, generate in an exclusive staging directory and validate before publication. Keep baseline outputs unchanged. A new current manifest may intentionally mark independent sections unavailable with reasons; critical model/policy incompatibility disables serving. A failure must not replace the current pointer with an incomplete pack. `--check` performs no fits, predicts or writes.

The manifest includes copied/derived evidence and explicit legacy references. A checksum proves file identity, not historical methodology: imported tuning/importance evidence carries inherited limitations. Unsupported legacy Train_* fields are never adopted just because their names exist. New train and validation scores come from the same newly fitted fold models and are shown as supplemental evidence, not relabelled historical scores.

### Frozen supplemental evaluation

Before opening test targets, freeze candidate estimator parameters, selected-full identity, Top-2 settings, features, DEV-fold membership, seed and no-selection policy. This declaration is an audit boundary, not a claim the old test has never been seen.

- Candidate evidence: five factory candidates on identical DEV folds, train+validation metrics from each fitted pipeline, runtime and dimensions, RF importance captured during those same fits. Compute comparison means and population SD (`ddof=0`) from fold rows.
- Final feature variants: evaluate the saved full configuration and fixed Top-2 configuration on the same folds. No hyperparameter search, new feature selection or promotion gate. Fit Top-2 on all DEV and serialize; load/reuse the full bundle and verify settings and stored predictions.
- Test evidence: frozen candidates each receive an all-DEV fit for their historical test metrics; Top-2 and full share the entire test population. Full/Top-2 residuals are `actual - predicted`; audit variance is separately `predicted - actual`. Empirical q90 is the 0.9 quantile of test absolute errors using the current default linear method. Same-data coverage is descriptive, not calibration validation.
- Importance: encode from the model and raw permutation MAE increase with 12 repeats/seed, reuse verified full evidence where valid. Store model/configuration IDs for every table. No claim that importance is a causal lever.
- Benchmark examples: first up to three strictly eligible test rows by stable source-row order; choose without targets, predictions or errors. Attach real actuals and full/Top-2 predictions after selection. Default fixtures and real generation do not need exception examples; explicit exception tests cover the allowed alternate path.
- Tuning: import/validate the existing five sweep tables with actual parameters, stage identity and historical provenance. Show applied values from the bundle, not the summary's unsupported explanation strings. If missing or conflicting, the affected tuning section is unavailable. Do not rerun a non-nested search to fill presentation data; fresh tuning is outside scope.

Generated summary/comparison/test/importance charts are saved as Plotly HTML under the supplemental pack using the same figure builders as the UI (no new image-export dependency). JSON/CSV remain authoritative. Paths are cited in the guide/verification record.

### Role constraints, exceptions and queue state

`scenario_policy.py` owns one policy implementation shared by offline example selection, manual entry, benchmark lookup, growth generation and pre-inference checks. DEV pairs are intersected with compatible metadata enums; bounds/median are per title across DEV categories. Show pair/title support counts. No target-derived policy and no test-informed fitting.

Manual scenarios always use strict observed pair and title bounds. Quick-load/curve exceptions are off by default; only experience outside title bounds but within 0–15 may be acknowledged. Benchmark input must still match an immutable manifest record, not browser-supplied actuals or changed years. Unobserved pairs, unknown categories/titles, invalid numbers, missing metadata and mismatched models always fail. Exceptions carry a fixed reason, source type and evidence/policy-bound acknowledgement, warning labels and no guaranteed coverage. Disabling the exception opt-in clears/revalidates affected scenarios/results.

Use session-local queue identities and a monotonically increasing revision. Binding is `(resolved_workspace, evidence_id, policy_hash, queue_revision)`. Changing workspace/evidence clears old queue/results/acknowledgements; adding or clearing invalidates prior results; reruns without change retain them. Validate all rows before one batch `predict`; failure yields no partial new result. Manual actuals cannot be supplied; benchmark actuals come from verified evidence only. Duplicate scenarios retain distinct IDs. Queue cap is 100; empty queue disables Run.

Growth view defaults to strict per-title domains for active queue profiles (same-category curves may overlap because title is only a label). Grid contains integer years within bounds and exact endpoints; optional extension to 0–15 is visually dashed and labelled. Compute curves only on explicit request from the successful queue snapshot; validate derived points before prediction. A curve is not a causal or monotonic salary guarantee.

### Page organization and presentation

Page 04: five KPIs; exactly four tabs. Overall comparison: labelled train/validation combo, ratio/R² combo, executive table, actual fold timeline. RF: fold combo, drift, concise actual-configuration chips. GB: head-to-head fold chart and MedAE comparison. Baselines: Linear collapse contrast and Linear/Ridge/nonlinear comparison. Evidence downloads remain in collapsed sections within these tabs. Diagnostic status uses the precise rules in the UI contract, not invented overfit thresholds.

Page 06 (P1, next): four blocks with reactive selectboxes and experience slider outside a form, explicit strict/exception status, queue and controls, chart-first results, optional growth view, audit and source-schema downloads. Every model chip comes from Top-2 metadata. Missing Top-2 does not invoke full-model fallback. Figures show 10 rows per page with a declared page range; table/export retain all rows. Actual markers may lie outside the interval.

Page 05 (P2): six KPIs and five sections in stakeholder order. Dollar metrics use their own scoreboard chart; R² uses a separate unitless panel. Residuals/actuals use the matching full-model historical test evidence. Methodology is collapsed by default. Four sensitivity combos show actual sweep tables and distinguish stage best from applied parameters. Feature-set comparison uses the new matched-population full/Top-2 pack, not manually entered metrics. Trust section labels historical examples and same-population q90 coverage honestly.

Use native Streamlit containers, accessible text labels and short captions. Retain existing Plotly (no chart-library migration) and routing. Cache data/resource loads by workspace plus verified immutable evidence/bundle identity; never cache mutable per-session queues. Missing chart inputs produce a local unavailable message with a CLI remediation command and do not hide unrelated valid sections.

#### Visual-priority amendment

Centralize semantic colors and chart layout metadata in `src/pages/model_evidence.py` so role colors do not drift between page-local figures. Add small helpers/constants rather than custom CSS or a new component layer. Figure tests assert trace colors by semantic role and confirm legends/line styles/marker shapes provide redundant encoding.

Assign every chart P1/P2/P3 according to `contracts/scenario-ui.md`. Keep `show_plot(..., width="stretch")`, but stretch only within the chart's assigned native Streamlit container. P1 charts use the main content width. P2 charts use bounded `st.container(width=...)` or a two-column row. P3 diagnostics and four tuning sensitivities use compact paired layouts/expanders. Responsive native stacking remains acceptable; no fixed width may cause horizontal page scrolling.

Page-specific changes:

1. Page 04: blue/teal validation columns, orange train line, dashed dark-red Dummy floor; P1 overall comparison full width, ratio/R² bounded P2, timeline/drift compact P3.
2. Page 05: historical-test errors are red columns and DEV-CV is an orange line in the primary dollar-error combo; R² is compact P2. Actual/predicted remains readable; residual/ECDF/encoded importance are supporting layouts. Four tuning charts render as a two-column P2 grid where labels fit.
3. Page 06: blue/teal prediction bars, amber uncertainty whiskers and diamond red actual markers; prediction chart is P1 and requested growth chart P2.

No scientific metric, artifact, evidence producer, inference result or export contract changes. Because producer dependencies include shared offline figure builders, implementation must run `ui_evidence --check`; regenerate the versioned supplemental chart pack only if the producing fingerprint genuinely changes.

#### Readability and third-party comprehension amendment

The root presentation issue is not solved by color or width alone. `common.show_plot` currently forces one narrow margin profile over every figure; compact charts still label every mark; conclusions primarily state caveats rather than the report's observed answer. The detailed runtime/source audit is in `ui-readability-audit.md`.

Implementation remains surgical:

1. Extend pure view builders in `model_evidence.py` with figure-specific readability profiles and deterministic conclusion builders. Do not add an abstraction framework: small functions per page/section are preferred.
2. Change `common.show_plot` to merge safe defaults without overwriting explicit figure margins/heights/axis settings. Existing untouched page behavior must be regression-tested because this helper is shared.
3. Add a reusable native Streamlit conclusion renderer with four short fields (`Finding`, `Why it matters`, `Limit`, `Decision/use`) and no raw HTML requirement. Educational glossary text can be static; scientific takeaways must be derived.
4. Page 04 adds a novice page brief, glossary, section questions and conclusions for ranking/baseline/folds. Reduce direct labels on drift and dense fold charts; reserve sufficient axis margins and expose exact values in formatted tables.
5. Page 05 adds a brief/glossary and conclusions for CV/test deltas, diagnostics, tuning, variant reliance and uncertainty. Stack tuning charts when two-column geometry cannot meet the minimum profile; format scorecard/variant/audit tables with reader-facing labels and units.
6. Page 06 explains the active two-input domain before controls and adds a post-run batch conclusion bound to the current snapshot. Use short scenario IDs on charts and put full role details in hover/table. Preserve queue, prediction and export contracts.
7. Add automated geometry/label-density/conclusion tests before implementation. Runtime AppTest verifies ordering and source binding; Chrome DevTools screenshots/DOM bounding boxes at 1280×800 and 1440×900 remain mandatory visual acceptance. If MCP remains unavailable, stop with pixel acceptance pending rather than infer success from source.

No metric, model, policy, scenario validation or export calculation changes. Figure-builder edits may legitimately change the supplemental producer fingerprint and generated HTML references; if so, regenerate one versioned supplemental pack after all tests pass, then prove unchanged reuse. Never run the root pipeline for presentation remediation.

#### Table consistency and training-method amendment

Pages 04–05 add a shared, explicit table grammar without CSS. Small decision summaries use `st.table`, whose installed API renders Markdown and supports richer static styling; wide or complete evidence remains in `st.dataframe` with `column_config`. A legend defines bold as the governing result/selected value and italics as reference/context. Shared pure builders own reader-facing names, whole-USD/three-decimal-R²/one-decimal-percent/three-decimal-second formatting and deterministic highlights from unrounded values. Ties are labelled rather than resolved by rounded display order. Exports remain raw CSV evidence.

Page 04 places `How temporal K-fold validation works` after the metric guide and before the tabs, collapsed by default but marked as the priority methodology guide. It must use `temporal_cv_splits` semantics exactly: stable chronological ordering, contiguous blocks, one-block training followed by the next validation block, final validation remainder, fold-local preprocessing/model fit and aggregation across effective folds. It explains what data/features/candidate pipelines are trained, how leakage is limited, why out-of-time validation is useful, which metrics rank models, and why the method is not randomized or expanding-window K-fold. Active `candidate_fold_metrics.csv` and `fold_membership.csv` supply all dynamic facts and downloads.

Page 05 expands its tuning guide before sensitivity charts. It derives stage counts/search spaces and trial winners from imported tuning tables, applied values from `manifest["models"]["full:selected"]["parameters"]`, and selection provenance from the validated inherited workflow. Copy explains the four-config anchor, sequential 6/6/4/6 coordinate sweeps, five temporal folds, descending CV R² ranking, fixed/carried values, no locked-test selection and no outer nested evaluation. It explicitly states that the final estimator comes from the first row of the R²-sorted initial table; later stages are sensitivity evidence. Unsupported legacy `tuning_rationale` is neither displayed nor paraphrased as fact.

Add a focused fixed-workspace, read-only presentation module at `src/pages/model_training_presentation.py` rather than changing the evidence-producing `model_evidence.py` or adding a service. It scans only `outputs/08_full_pipeline/logs/`, resolves complete manifests, matches the current raw-source SHA-256, verifies relevant export/log checksums and path containment, and returns a small descriptor plus downloadable bytes/tables. The UI labels audit run IDs and historical-primary-pipeline scope; active supplemental evidence remains authoritative and no values are blended. Isolating this consumer-only code preserves the active pack's producer fingerprint, so missing/incompatible logs are optional unavailable states and no model training, evidence-pack regeneration, dependency change or modification to the protected primary pipeline is required.

### Exports and compatibility

Lock both CSV contracts before code (see [scenario-ui.md](contracts/scenario-ui.md)). The new audit export has explicit schema version, scenario/record IDs, feature variant, evidence/model identity, validation mode, exception reason, point/bounds/actuals, absolute error and signed variance. Preserve full precision for arithmetic; export money to cents, percentages to four decimals, visible chart labels to whole USD/one decimal percent. Blank means unavailable, never zero by default.

Original-source export uses the exact 25 `SOURCE_COLUMNS`, including `AI Engineering`. Manual unknown columns remain empty, including target and posting dates. Synthetic IDs have a `SCENARIO-` prefix. Benchmark source values are preserved only when known from verified evidence; do not invent original `job_id` lost during cleaning. Audit/source filenames differ and the UI explains the source-schema file is not a completed training dataset. The old `error_pct` name is not silently repurposed.

## Phase 2 — Verification and Delivery Plan

1. Record explicit task approval, preserve the dirty working tree and baseline hashes, re-query Graphify, and verify source/helper/dependency assumptions before coding.
2. Build manifest/policy/evaluator tests first with isolated small fixtures and fake estimators. Verify failure before implementation. Contracts include corrupt/missing packs, path escape, partial publication, feature mismatch, DEV/test leakage, unknown pairs, fractional medians, nonfinite values and immutable benchmark identity.
3. Implement the new producer in focused files; use temporary-workspace generation to prove determinism and per-component reuse. Tests may fit controlled tiny fixtures; no fixture run mutates baseline artifacts.
4. Deliver Page 04 MVP using verified fixture packs, then Page 06, then Page 05. Test each independently through the established renderer and actual `streamlit.py` entrypoint with controlled login/data-root fixtures. Keep the global upload/process action outside new page tests and document its inherited issue.
5. After producer tests pass, generate the actual missing/invalidated supplemental pack with the new CLI only. This is necessary new-artifact generation, not a demo rerun. Run `--check`, read/predict-only consumer tests and an unchanged producer invocation with fit spies proving reuse. Reconfirm baseline hashes and `core.py` unchanged.
6. Run relevant existing regression tests plus the full pytest suite; any unrelated producer-heavy tests run only in their documented isolated workspaces. Do not execute root `pipeline.py` to satisfy UI review. Record exactly what was tested and any inherited failures; never label an unrun test passed.
7. Exercise the real `streamlit.py` entrypoint in a browser at 1280/1440 widths: all pages/roles, chart labels, focus/keyboard operation, 1/3/100 scenarios, page slices, actual outside band, exception toggles, switching evidence and CSV downloads. Record warm/cold timings and reviewer walkthrough outcomes. If browser tooling is unavailable, leave visual acceptance pending for human review rather than claim it passed.
8. Update `docs/MODEL_UI.md`, README and `verification.md`, align living specs/tasks, run Graphify update or document a valid reason, and inspect the final diff before requesting merge review.
9. For the visual-priority amendment, add failing role-color and layout-priority tests first, implement the shared palette and bounded native containers, run AppTest at both target widths where tooling permits, and record screenshots or leave pixel acceptance explicitly pending. Do not regenerate model evidence solely for color/width changes.
10. For readability remediation, capture failing geometry/label-density/conclusion tests, fix shared margin merging and page-specific profiles, then perform a third-party comprehension walkthrough. Regenerate supplemental evidence only when the shared generated-chart fingerprint changes; record the new evidence identity and protected-hash parity.
11. For the table/training-method amendment, first test pure table builders, highlight/tie rules, displayed precision, actual fold/tuning narratives and audit-log compatibility/path/checksum rejection. Then update Pages 04–05, preserving chart-first scientific sections while putting the two collapsed methodology guides before their detailed evidence. Verify download bytes against source files and assert zero fit/tune/producer calls.
12. Run focused AppTest/full pytest/Ruff/diff checks and `ui_evidence --check`. Because this amendment only reads existing evidence and does not change a producer or artifact contract, reuse `ui-b35971c9f8bcd166`; do not train or regenerate unless implementation unexpectedly changes a producing fingerprint, in which case stop and explicitly invalidate/regenerate the affected pack before reliance.

Commands and expected checks are in [quickstart.md](quickstart.md). Task execution is gated; no tasks are marked complete by this plan.

## Complexity Tracking

| Violation / bounded exception | Why needed | Simpler alternative rejected because |
| --- | --- | --- |
| Historical non-nested tuning and R² tuning vs MAE family ranking (VI.3/VI.4) | Display existing experiment honestly; Top-2 uses fixed inherited settings and no new search/promotion | Rewriting tuning/selection changes the primary experiment beyond the approved missing-evidence scope; hiding the discrepancy is not acceptable |
| Previously exposed test population | Users need reproducible historical comparisons and benchmark scenarios, not an invented pristine split | Retroactive 295+3 splitting cannot undo exposure; new external data collection is outside scope. Never waive leakage safety or claim independent validation |
| Existing global upload handler fits in Streamlit | Preserve unrelated app routing/upload behavior; explicitly limit this feature to new offline command and fit-free Page 04–06 actions | Reworking the entire upload lifecycle is a separate architecture feature. This inherited issue remains a disclosed risk, not a precedent for new UI training |
| Supplemental report namespace rather than overwrite | Preserve current producers/Pages 07–08 and establish explicit model/evidence identities | In-place overwrite silently changes public old artifacts; complete main-pipeline regeneration performs unrelated expensive training |

These are bounded design treatments for task review, not a declaration that historical governance issues are fixed. New data leakage, fabricated provenance or invalid input cannot be waived. Any requirement for fresh nested tuning, genuinely untouched data or primary-model replacement requires a separate scope revision.
