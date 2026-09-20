# Research: Show Offline Training Validation on Pages 04–05

**Date**: 2026-09-20  
**Scope**: Repository evidence and Streamlit presentation design only; no fitting, artifact generation, or UI edits.

## Evidence reviewed

- `src/pages/page04_model_comparison.py` and `page05_best_model.py` currently consume `outputs/ui_evidence/current.json` through `model_evidence.load_evidence()` and return early when that source is unavailable.
- `src/pages/model_evidence.py` contains shared colors, conclusion rendering and existing supplemental-evidence loading; it also imports joblib for Page 06 serving, so extending that module would unnecessarily mix the new no-model-load consumer with serving concerns.
- `src/ai_job_market/training_evidence_io.py::validate_complete_pack()` validates immutable `training-validation/v1` packs, required files and external bundle hash/metadata without deserializing bundles.
- Training-validation packs have no mutable `current.json`; final directories are `outputs/training_validation/tv-<32 hex>/`, with `generated_at` in `manifest.json`.
- Existing Pages 04–05 already contain extensive primary supplemental evidence. The new source must be visually and scientifically separate, not merged into current rankings or historical-test claims.
- Real-entrypoint AppTest already covers both pages in `tests/test_model_ui_pages.py`; pure evidence/view behavior is tested in `tests/test_model_evidence_views.py` and can be extended with a focused new module test.
- Graphify query `Page 04 Page 05 evidence loaders expanders unavailable state training validation ui_summary` found the narrow dependency path through the two pages, `model_evidence.py`, `ui_evidence_io.py`, `training_evidence_io.py`, and `build_ui_summary()`.

## Decisions

### D1. Add one focused read-only presentation module

**Decision**: Add `src/pages/training_validation_presentation.py` for discovery, validation, bounded loading, formatting and expander rendering data. Pages 04–05 call it near the top of `render()`.

**Why**: This avoids coupling training-validation consumption to Page 06's joblib-serving helpers and keeps the producer unchanged.

**Rejected**:
- Extending `model_evidence.py`: mixes two schemas and introduces a misleading dependency on serving/model-loading code.
- Updating `ui_evidence/current.json`: blends independent experiments and mutates an existing producer contract.
- Creating a generic evidence plugin framework: unnecessary for two pages and one pack type.

### D2. Select the latest valid immutable pack, not the latest directory

**Decision**: Scan at most 100 `tv-*` final directories, reject symlinks and malformed IDs, parse bounded manifests, validate each with `validate_complete_pack()`, and select the greatest valid UTC `generated_at`; tie-break by run ID. Keep bounded reason codes for invalid candidates.

**Why**: A corrupt newer pack must not hide an older complete pack. Directory mtime is mutable and scientifically meaningless.

**Rejected**:
- Lexicographic run ID: IDs are experiment hashes, not chronology.
- Filesystem mtime: mutable and not part of the evidence contract.
- New `current.json`: requires a producer change and mutable pointer not requested by the user.

### D3. Render before existing source early returns

**Decision**: Render the training-validation expander immediately after each page title/caption and before loading existing supplemental evidence.

**Why**: Training-validation and supplemental UI evidence must fail independently. A missing source cannot suppress the other's status.

### D4. Use native collapsed expanders and bounded tables

**Decision**: Use native `st.expander(..., expanded=False)`, `st.table` for small conclusions/key-value evidence, and `st.dataframe` with reader-facing columns for bounded detailed tables. Do not add CSS, custom HTML, components, or new charts unless existing evidence cannot be understood as a table.

**Why**: The user asked to make offline evidence readable in expanders, not redesign the pages. Native elements are accessible, testable, and consistent with the app.

### D5. No artifact regeneration

**Decision**: Reuse existing complete packs. The feature changes only a consumer and does not change training code, source data, policy, dependency lock, artifact schema, or produced contents. Do not run real training or regenerate evidence for UI review.

**Current reality**: The current workspace has no eligible real pack, so its expected UI is `Training validation unavailable`. Tests may construct complete fixture packs in temporary workspaces only.

### D6. Fail closed without fallback

**Decision**: Missing/corrupt/partial sections show stable local unavailable reasons and the safe read-only `inspect` command. The UI never pulls values from fixture files, `ui_evidence`, historical audit logs, old pipeline outputs, or hardcoded examples.

### D7. Keep expensive or unsafe work out of Streamlit

**Decision**: The consumer may hash/read bounded local evidence files but must not import/call training orchestration, fit/search/predict/benchmark/publication/holdout-access functions or joblib. Tests patch these boundaries and assert zero calls.

### D8. Build a detailed evidence-derived transcript; do not mislabel the short raw log

**Decision**: Preserve and download `training.log`, `events.jsonl`, `report.md`, `agent_summary.json` and `model_conclusions.json` exactly. Separately construct a deterministic `evidence-derived transcript` from fold summaries/memberships, candidate fold metrics, tuning trials/context summaries, variant folds, holdout metrics, explainability/subgroup/uncertainty evidence and conclusions. Show a bounded preview and generate the complete transcript CSV in memory for download without writing to the workspace.

**Why**: The current `training.log` is a short completion summary and current `events.jsonl` is lifecycle-oriented. The authoritative tables contain the exact row counts, folds, parameters and model outcomes needed to answer the user's question. Labelling the reconstruction explicitly is more honest than presenting derived steps as raw runtime events or changing the offline producer in this UI-only feature.

**Transcript invariants**:
- method order is deterministic and does not imply timestamps absent from evidence;
- no scientific metric is recalculated except display-only grouping/aggregation explicitly identified by method;
- each row names its source file/reference;
- all 25 candidate-fold evaluations and every model conclusion remain visible/downloadable;
- every outer fold answers parent/train/validation/later/added/protected counts;
- tuning run/skip/reuse and Full/Top-2/frozen holdout steps remain distinct.

### D9. Reuse the completed historical Branch B audit instead of retraining

**Decision**: Add a separately labelled historical training report from the active validated supplemental pack plus the newest complete source-compatible primary-pipeline audit already under `outputs/08_full_pipeline/logs/`. Do not run the root pipeline or attempt to convert historically exposed rows into a strict `training-validation/v1` pack.

**Evidence**: The current audit manifest reports `training_status=completed`, complete log/console/coverage, matching raw-source SHA-256, fold membership, model-comparison/tuning exports, and a complete JSONL trace. Existing `outputs/04_model_comparison` and `outputs/05_best_model` plus the active `ui_evidence` pack already contain the metrics/configuration evidence needed for report display. The repository rule requires reuse when compatible artifacts exist and producer contracts are unchanged.

**Branch boundary**: The common prepared feature base produced before branch divergence supports both Branch A segmentation and Branch B salary regression. Branch B may run after upstream/Branch A stages in the pipeline, but cluster assignments are not approved salary-model features and will not be silently introduced. The report describes this sequence rather than changing model inputs.

**Scientific label**: This source is `historical primary-pipeline + supplemental evidence`, not pristine future validation. March-2026 locked-test exposure remains explicit. Strict training-validation unavailability remains visible and independent.

### D10. Evidence-backed 5W1H rather than narrative invention

**Decision**: Build deterministic 5W1H rows for each candidate and selected/final role:

- **Who**: pipeline/audit run and model role/configuration identity;
- **What**: continuous salary regression target and validated feature/configuration scope;
- **When**: actual fold/evaluation periods from evidence;
- **Where**: DEV temporal validation or historically scored locked-test scope;
- **Why**: declared comparison role and evidence-backed decision only;
- **How**: implemented fold-local pipeline, estimator configuration, metrics and timing;
- plus result, limitation, next action and source references.

Unsupported causal reasons from `docs/spec-imporve-ui.md` remain illustrative and are not copied as facts. In particular, covariance singularity, memorization, economic causes and “pristine” test claims require evidence not present in the current artifacts.

### D11. Additive visual hierarchy and log treatment

**Decision**: Preserve every existing Page 04/05 section. Add compact highlighted metric/status strips and collapsed per-model 5W1H/log details within the focused presentation layer. Follow the referenced design's chart-first, direct-label, minimal-prose/high-density intent, but do not add duplicate charts when existing charts already carry the metric story.

Raw JSONL events remain raw and bounded in preview; complete bytes and manifest/export downloads remain checksum verified. Any reshaped model timeline/report is labelled evidence-derived and names its source event/export.

## Streamlit guidance applied

- Native `st.expander` is appropriate for optional detailed evidence.
- No expensive computation is triggered by expander state; validation is bounded and read-only.
- Use `width="stretch"`, never deprecated `use_container_width`.
- Use `st.dataframe` column configuration for currency/percentage/seconds and `st.table` only for small static summaries.
- Use AppTest from the real `streamlit.py` entrypoint for page behavior; pure discovery/formatting receives ordinary pytest coverage.

## Risks

1. Hash-validating many packs on every rerun can be expensive. Bound discovery to 100 and keep presentation projections small; cache only if tests show a measurable need and cache identity includes workspace/run/manifest metadata.
2. Existing `validate_complete_pack()` validates all required files and bundle metadata, which is intentionally stricter than reading only `ui_summary.json`; partial packs therefore remain unavailable rather than partially trusted.
3. `generated_at` must be strict UTC. Malformed timestamps invalidate that candidate for latest selection.
4. Browser visual acceptance may be unavailable; leave it pending rather than infer layout from source.
