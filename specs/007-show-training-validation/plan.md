# Implementation Plan: Show Offline Training Validation on Pages 04–05

**Branch**: `007-show-training-validation` | **Date**: 2026-09-20 | **Spec**: [spec.md](spec.md)  
**Input**: Feature specification from `specs/007-show-training-validation/spec.md`  
**Status**: Ready for task review; no implementation approval yet.

## Summary

Add a small read-only consumer for immutable `training-validation/v1` packs and render an additive always-visible trust overview plus Page 04 fold/candidate evidence and Page 05 tuning/final evidence in native collapsed Streamlit expanders, including a detailed evidence-derived step transcript and complete original log/report downloads. Select the latest valid UTC-dated final pack deterministically. If no compatible pack exists, show a truthful actionable unavailable expander and continue rendering existing page content. Do not edit the training producer, primary pipeline, Page 06, model bundles, source/prepared data, or dependency lock; do not retrain or regenerate artifacts.

## Technical Context

**Language/Version**: Python 3.13.12; Ruff target py312  
**Primary Dependencies**: Existing Streamlit 1.64.0, pandas 2.3.3, Plotly 7.1.0, standard-library pathlib/json/datetime; existing training-validation and UI evidence readers  
**Storage**: Read-only immutable files under `outputs/training_validation/<run_id>/` plus validated external bundle metadata under `artifacts/training_validation/<run_id>/`  
**Testing**: pytest 9.1.1, pure contract/view tests, `streamlit.testing.v1.AppTest` through `streamlit.py`, existing full regression suite  
**Target Platform**: Existing Linux offline workspace and Streamlit desktop web UI  
**Project Type**: Offline ML evidence application with Streamlit artifact consumers  
**Performance Goals**: Bound discovery to 100 final run directories; bounded display projections; no model deserialization or model operation; ordinary Page 04/05 reruns remain interactive  
**Constraints**: One native always-visible overview plus native collapsed detail expanders, no CSS/custom components/dependency additions; preserve all existing Page 04/05 analytical sections; no `core.py`, producer, pipeline, Page 06, preprocessing, artifact, pointer, or serving changes  
**Scale/Scope**: Two existing pages; one new focused presentation module; approximately 42 files per complete pack, 5 outer and 18 inner fold descriptors, 25 candidate-fold rows, up to 156 tuning slots, 10 matched variant-fold rows, 5 candidate and 2 final roles

## Constitution Check

### Pre-research gate

| Gate | Disposition |
| --- | --- |
| Data integrity | Pass. Read-only summary consumption; no source rows, fitting, target access, or partition mutation. CV/holdout/reserve roles remain labelled. |
| Reproducibility and ML evidence | Pass. Existing complete packs are reused because producer code/data/config/dependencies/schema are unchanged. No retraining or regeneration. |
| Verification | Pass by design. Failing-first loader/view tests, corruption tests, no-operation spies, real-entrypoint AppTest, full regression, Ruff/diff and protected hashes are planned. |
| Streamlit boundary | Pass. UI validates and renders files only; no run/fit/tune/predict/load/publish calls. |
| Input validation | Pass by design. No new user path/run selector. Fixed workspace discovery validates paths, run IDs, UTC timestamps, schemas, hashes, sizes and table contracts with bounds. |
| Scientific/security claims | Pass. Sources stay separate; CV, holdout, historical test, q90, scientific and operational states are labelled. Errors are bounded and sanitized. |
| Documentation | Pass by design. Update model UI guide/README and document source boundaries and unavailable behavior. |
| Graphify/Karpathy | Pass. Query identified a surgical two-page/one-module path. No plugin framework, pointer, model loader or pipeline integration. |

### Post-design gate

All gates remain passed. No constitutional exception is required. The consumer contract is additive and does not change training artifacts; therefore existing complete packs may be reused. The current workspace has no pack and must display unavailable rather than trigger generation.

## Project Structure

### Documentation

```text
specs/007-show-training-validation/
├── spec.md
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   └── training-validation-ui.md
├── checklists/
│   └── requirements.md
└── tasks.md
.specify/assessments/show-training-validation/research.md
```

### Source Code

```text
src/pages/
├── training_validation_presentation.py   # NEW read-only discovery, validation, projections, renderer
├── page04_model_comparison.py            # small expander call before existing evidence early return
└── page05_best_model.py                  # small expander call before existing evidence early return

src/ai_job_market/
├── training_evidence_io.py               # REUSE unchanged complete-pack validator
└── training_*.py                         # UNCHANGED offline producer/search/evaluation/runtime/report

pipeline.py                               # UNCHANGED
src/pages/page06_prediction.py            # UNCHANGED
src/ai_job_market/ui_evidence_io.py       # UNCHANGED independent supplemental source

tests/
├── test_training_validation_views.py     # NEW pure discovery/schema/projection/render-data tests
├── test_model_ui_pages.py                # extend real-entrypoint expander/unavailable/no-operation tests
└── test_model_evidence_views.py           # only if shared reader-facing table assertions belong here

docs/MODEL_UI.md                          # document second evidence source and unavailable state
README.md                                 # concise guide link/behavior update
```

**Structure Decision**: One new consumer-only module keeps pack selection/validation and view formatting independently testable. Two page files receive only small render calls. The existing model evidence module is not extended because it also owns Page 06 bundle loading and uses a different artifact schema.

## Phase 0 — Research Decisions

See [research.md](research.md). Key decisions:

1. Select latest **valid** pack by strict UTC `generated_at`, tie by run ID.
2. Validate with `validate_complete_pack()`; never deserialize model bundles.
3. Render before existing supplemental-source early returns.
4. Use one removable native trust-overview container plus native collapsed detail expanders and bounded reader-facing tables/downloads.
5. Preserve original log/report files and construct a separately labelled evidence-derived transcript from authoritative tables; never call it raw execution telemetry.
6. Reuse existing artifacts; do not run the offline producer for this UI-only change.
7. Keep unavailable state explicit and prohibit all fallback data.

## Phase 1 — Design

### Discovery and validation flow

1. Resolve the established workspace root.
2. Inspect fixed `outputs/training_validation/` only; if absent, return `not_found`.
3. Enumerate at most 100 non-symlink directories matching `tv-[0-9a-f]{32}`. Ignore hidden staging and `holdout_access` by pattern.
4. Read each `manifest.json` under the existing control-file size bound, parse strict UTC `generated_at`, and call `validate_complete_pack(root, run_id)`.
5. Record bounded stable reason codes for invalid candidates without exposing raw content.
6. Select max `(generated_at, run_id)` among valid candidates.
7. Load only manifest-listed CSV/JSON/text evidence through safe paths; validate table schema, finite/null semantics, role counts, partition labels, run/source consistency and row bounds.
8. Reconcile raw `training.log`/`events.jsonl` with authoritative tables where identities overlap; disagreement creates an explicit warning, never a silent favorable choice.
9. Produce method-ordered `TrainingTranscriptRow` records covering partition/folds, 25 candidate-fold evaluations, model conclusions, tuning contexts/trials, variants, holdout, explainability, subgroup, uncertainty and publication evidence. Every row carries a source reference and is labelled evidence-derived.
10. Produce `TrainingValidationView` with independent Page 04/Page 05 sections, bounded transcript previews, complete in-memory transcript CSV and original-file download descriptors.

### Page 04 presentation

A compact always-visible `Training & validation evidence` container precedes the detail expander. It shows verified run/count/trust context when available and an explicit non-evidence capability summary when unavailable; it does not modify existing analytics.

A collapsed expander labelled `Latest training validation: folds and candidate models` appears after the page title/caption and before existing supplemental evidence loading. Available content order:

1. run/status/source caption and explicit distinction from the existing supplemental experiment;
2. partition counts/shares and five outer fold method/count table;
3. parent-labelled inner fold table and monthly counts in a secondary bounded section;
4. candidate summary with MAE/RMSE/MedAE/R²/runtime;
5. selection summary separating lowest CV MAE, selected family and fastest fit;
6. fit/runtime/operational context and five candidate conclusions;
7. a method-ordered transcript preview containing all five outer split explanations and 25 candidate-fold evaluations;
8. five candidate conclusions with finding, decision, limitation and action;
9. byte-identical original log/report/summary downloads and a complete generated transcript CSV.

No new candidate chart is required; the user requested readable expander evidence, and tables avoid duplicating existing page charts.

### Page 05 presentation

The same compact always-visible trust container precedes the detail expanders and uses Page 05 counts for tuning contexts, variant folds, holdout roles, and final conclusions.

Two collapsed expanders avoid one oversized panel:

- `Latest training validation: tuning and Full/Top-2 variants`
- `Latest training validation: holdout, explainability and uncertainty`

They show tuning status/context/winner/fit count; parent-labelled evidence references; matched variant fold metrics; one-time holdout metrics/prediction summary; encoded/permutation importance method labels; subgroup support/small-sample flags; descriptive q90/coverage caveat; scientific and operational states; final conclusions and downloads. A method-ordered transcript preview includes every tuning context/trial state, variant step and final evaluation step, with the complete transcript downloadable. Reserve exclusion is always visible.

### Unavailable and partial states

The same expander labels remain present. With no selected pack, each contains:

- `Training validation unavailable` warning;
- stable safe reason;
- statement that no fallback values are displayed;
- read-only `inspect` command;
- explanation that `run` requires eligible future data and exact approval.

A valid pack with an optional section unavailable shows that section's reason locally; core manifest/table inconsistency invalidates the candidate pack.

### Performance and safety

- No joblib import in the new module.
- Transcript construction is pure reshaping/formatting over validated values; it does not recompute model selection or scientific metrics. Derived display aggregations are method-labelled.
- Preview rows are capped at 200 per table, while generated transcript download bytes include every deterministic row and are never written to disk.
- No user-supplied path/run selector.
- Display tables are projected to approved columns and capped; complete source files remain downloadable.
- Do not hide sensitive fields via column configuration alone; never include raw row payloads in view projections.
- Loader errors map to stable reason codes/messages; no traceback or raw JSON is rendered.
- Add caching only if measured AppTest/runtime evidence warrants it; simplest initial implementation remains uncached and bounded.

## Historical Branch B 5W1H amendment

Reuse the existing active supplemental evidence loader and extend the compatible-audit reader with bounded validated event projections; do not introduce another producer or run training. Add pure builders for candidate/final `Model5W1HRecord` rows in the focused presentation layer. Page 04 receives a compact highlighted historical-evidence strip plus five collapsed candidate records and audit-log access. Page 05 receives selected-family/Full/Top-2 records, tuning/final scope and the same run-bound log access. Existing charts, tabs and KPIs are not moved or modified.

The report follows `docs/spec-imporve-ui.md` through native metric/status chips, direct labels, concise text and detail-on-demand. Values come from active evidence and audit exports, not illustrative numbers in that document. The strict validation overview remains first and continues to report unavailable for the exposed current reserve.

The current complete audit and active supplemental pack are reusable inputs. Root pipeline execution, artifact regeneration, model loading, fit/predict and changes to Branch A/Branch B feature policy are explicitly excluded.

## Artifact Reuse / Invalidation Decision

No training-producing code, source/prepared data, config, dependency lock, training-validation schema or artifact contents change. Existing complete packs remain valid and MUST be reused. No pack currently exists in the real workspace, so implementation verification uses temporary fixture packs and the real workspace shows unavailable. The offline producer MUST NOT be run merely for UI review. If implementation unexpectedly requires a producer/schema change, stop, revise this plan/tasks, explicitly invalidate affected packs and obtain approval before regeneration.

## Verification Plan

1. Capture protected hashes for producer modules, pipeline, Page 06, config, outputs/artifacts and inherited `uv.lock`.
2. RED pure tests for no directory, staging-only, malformed IDs/timestamps, symlinks, checksum corruption, newer-invalid/older-valid selection, tie behavior, table contract/count/partition failures and bounded scan.
3. RED transcript tests reconciling every fold/count, 25 candidate-fold rows, five candidate conclusions, tuning contexts/trials, variants, holdout roles, final conclusions, source references, deterministic ordering and original/generated download completeness.
4. Implement minimal consumer/transcript and make pure tests GREEN.
5. RED/GREEN Page 04 AppTest for available/unavailable expander, exact fold 3 evidence, five conclusions, selected/fastest distinction, transcript preview/downloads, and independence from existing supplemental evidence failure.
6. RED/GREEN Page 05 AppTest for tuning skip/run, variants, holdout labels, reserve exclusion, q90 caveat, subgroup flags, final conclusions, transcript preview/downloads, and source parity.
7. Spy on fit/search/predict/benchmark/joblib/publication/holdout/pipeline boundaries during page reruns and assert zero calls/writes.
8. Run affected tests, full pytest, Ruff and `git diff --check`; compare protected hashes.
9. Exercise real `streamlit.py`; expected current workspace state is unavailable. Use browser tooling for 1280×800 and 1440×900 visual checks if available, otherwise record pending human acceptance.
10. Update docs, Graphify code graph, living specs/tasks and verification evidence.

## Complexity Tracking

No constitutional violation or complexity exception is required.
