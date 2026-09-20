# Feature Specification: Show Offline Training Validation on Pages 04–05

**Feature Branch**: `007-show-training-validation`  
**Created**: 2026-09-20  
**Status**: Draft for task review; no UI implementation approved yet  
**Input**: Show the existing offline `training_validation` evidence on Streamlit Pages 04 and 05 inside expanders. Streamlit remains read-only. If no complete compatible pack exists, show an explicit unavailable state and never substitute fixture, historical, or fabricated evidence.

## User Scenarios & Testing

### User Story 1 - Understand folds and candidate evidence on Page 04 (Priority: P1)

A report reader opens Page 04 and expands the latest training-validation details to understand the approved partition, exact outer and inner fold months/row counts, five candidate results, fit diagnostics, runtime evidence, selection decision, and each candidate conclusion without reading raw offline files.

**Why this priority**: The user explicitly cannot read the offline pack today; fold construction and candidate comparison are the first decision layer.

**Independent Test**: With a valid fixture pack and existing supplemental UI evidence, open Page 04 through the real Streamlit entrypoint and verify the expander shows the pack identity, exact outer/inner rows and months, candidate metrics, runtime/selection distinctions, conclusions, and evidence downloads without any fit/predict/load call.

**Acceptance Scenarios**:

1. **Given** at least one complete compatible training-validation pack, **when** Page 04 opens and the training-validation expander is viewed, **then** it identifies the selected latest pack and shows authoritative fold and candidate evidence with reader-facing labels and units.
2. **Given** multiple complete compatible packs, **when** Page 04 opens, **then** the newest validated `generated_at` value is selected deterministically and its run ID is visible.
3. **Given** candidate evidence, **when** conclusions are shown, **then** lowest CV MAE, frozen selected family, fastest fit model, limitations, and safe next actions remain distinct.
4. **Given** fold/model evidence, **when** the detailed training transcript is expanded, **then** every outer fold shows its parent denominator, train/validation months and exact rows, added history, later-unused rows, protected-population exclusions, and each of the five models has a fold-by-fold result plus a scoped final conclusion.

---

### User Story 2 - Understand tuning and final evidence on Page 05 (Priority: P1)

A report reader opens Page 05 and expands the latest training-validation details to inspect conditional tuning, matched Full/Top-2 evidence, one-time holdout metrics, residual/uncertainty evidence, feature importance, subgroup support, operational assessment, and final conclusions.

**Why this priority**: Final model interpretation must be understandable without opening JSON/CSV artifacts and without blending this experiment with the historically exposed supplemental evidence already displayed on Page 05.

**Independent Test**: With a valid fixture pack, open Page 05 through the real entrypoint and verify the expanders display tuning status, final variants, holdout scope, importance method, subgroup support flags, descriptive uncertainty caveat, scientific outcome, operational status, and final conclusions from the same run.

**Acceptance Scenarios**:

1. **Given** a complete pack where RF tuning ran, **when** the tuning expander is viewed, **then** all six search contexts and parent-labelled three-fold evidence are identifiable and the final applied settings are distinguished from trial evidence.
2. **Given** a complete pack where RF tuning was skipped, **when** the tuning expander is viewed, **then** the declared skip reason is shown instead of an empty or favorable tuning claim.
3. **Given** holdout diagnostics, **when** Page 05 displays them, **then** they are labelled `EVALUATION_HOLDOUT`, reserve results are absent, q90 is labelled same-holdout descriptive/not calibrated, and Full/Top-2 conclusions cannot silently promote a variant.
4. **Given** tuning/final evidence, **when** the detailed transcript is viewed, **then** it shows each search context, stage/trial parameters, parent inner folds, status/reuse/skip result, applied settings, matched variant steps and frozen holdout step in chronological method order.

---

### User Story 3 - See a truthful unavailable state (Priority: P1)

A reader can still open Pages 04 and 05 when no complete compatible training-validation pack exists. Each page contains an explicit collapsed expander explaining that training validation is unavailable, why no fallback is shown, and which safe offline inspection command to run.

**Why this priority**: The current real workspace is blocked and has no eligible pack. A fail-closed state is therefore the initial production behavior, not an edge-only state.

**Independent Test**: Open both pages in a workspace with no pack, staging-only packs, corrupt manifests, missing files, mismatched checksums, or unsupported schema. Verify the rest of each existing page remains usable and the training-validation expander shows an actionable unavailable message with no fixture/historical substitution.

**Acceptance Scenarios**:

1. **Given** no complete pack, **when** either page opens, **then** a `Training validation unavailable` expander is present and existing page evidence continues rendering independently.
2. **Given** only corrupt or incomplete packs, **when** either page opens, **then** the UI reports a bounded safe reason, does not deserialize models, and does not expose untrusted file content as instructions.
3. **Given** the current blocked workspace, **when** the unavailable expander is viewed, **then** it recommends read-only `inspect` and states that eligible data plus approval are required before `run`.

### User Story 4 - Review the completed historical Branch B training story (Priority: P1)

A third-party reviewer sees a clearly highlighted, additive report on Pages 04–05 even when strict future-reserve validation is unavailable. The report uses the already completed source-compatible primary pipeline audit and active supplemental evidence to explain the Branch A/common-foundation → Branch B flow, every candidate model in 5W1H form, model decisions, tuning/final evidence, and actual structured training logs. It is labelled historical/exposed evidence and never represented as a pristine `training-validation/v1` run.

**Independent Test**: Open Pages 04–05 in the current workspace and verify that five candidate 5W1H records, selected/final variant records, run identities, evidence-backed metrics/limitations, bounded event-log previews, and byte-identical log downloads are visible while all existing charts/tabs remain unchanged.

**Acceptance Scenarios**:

1. **Given** the complete source-compatible audit and active supplemental pack, **when** Page 04 opens, **then** it highlights five candidate-model 5W1H records grounded in the existing candidate/fold/configuration evidence.
2. **Given** inherited tuning and historically exposed locked-test evidence, **when** Page 05 opens, **then** it highlights selected-family and Full/Top-2 5W1H records, tuning/final outcomes, and limitations without claiming untouched or pristine validation.
3. **Given** complete audit JSONL, **when** a reviewer opens the training activity log, **then** bounded model-relevant events identify operation/status/run/source while the complete original log remains downloadable byte-for-byte.
4. **Given** strict training validation remains blocked, **when** historical evidence is available, **then** the historical report remains visible as a separately labelled source and does not populate strict pack fields.

### Edge Cases

- No `outputs/training_validation/` directory, an empty directory, only hidden staging directories, or only the `holdout_access/` ledger.
- More than one complete pack; equal timestamps; malformed/missing timestamps; a newer corrupt pack beside an older valid pack.
- A valid-looking manifest with unknown schema, wrong run ID, path escape/symlink, missing CSV/JSON/chart, checksum mismatch, or corrupt bundle metadata.
- Required tables exist but contain missing columns, duplicate model/fold identities, nonfinite metrics, count mismatch, an unexpected partition, reserve predictions, or fewer/more than five candidate roles.
- R² is explicitly unavailable; tuning is skipped; runtime/operational limits are unavailable; subgroup support is small; conclusions contain unavailable status.
- Existing supplemental `ui_evidence` is missing while training validation is present, and vice versa. Each source must fail independently without hiding the other source's status.
- Large evidence tables: UI displays bounded previews/summary tables while downloads preserve authoritative files.
- `training.log` contains only a short summary or `events.jsonl` contains only lifecycle events: the UI builds a clearly labelled evidence-derived transcript from authoritative fold/metric/trial/conclusion tables and never claims those derived rows were raw runtime events.
- Event/log/table disagreement: show the discrepancy and source references; do not silently choose whichever value appears favorable.

## Requirements

### Functional Requirements

- **FR-001**: Pages 04 and 05 MUST contain clearly labelled, collapsed training-validation expanders rendered before any early return caused by the existing supplemental evidence source.
- **FR-002**: The UI MUST discover only immutable final namespaces matching `outputs/training_validation/tv-<32 lowercase hex>/manifest.json`; it MUST ignore staging and holdout-ledger directories.
- **FR-003**: The UI MUST validate a candidate pack through the existing `training-validation/v1` read-only checker, including required evidence files and bundle metadata/hash/size, without deserializing a model.
- **FR-004**: When multiple valid packs exist, the UI MUST choose the greatest valid UTC `generated_at`, breaking ties by run ID; invalid newer packs MUST NOT suppress an older valid pack and their bounded reasons MUST remain available for diagnostics.
- **FR-005**: Discovery MUST be bounded to at most 100 final run directories and MUST reject malformed timestamps, unexpected run IDs, symlinks, path escapes, unsupported schemas, corrupt files, and oversized display tables with a safe unavailable/partial state.
- **FR-006**: Page 04 MUST display pack identity/status, partition counts and actual shares, five outer folds, parent-labelled inner folds, exact train/validation/later-unused row counts and periods, monthly counts, candidate summary, fit diagnostics, runtime/accuracy-cost evidence, family selection, and five candidate conclusions.
- **FR-007**: Page 04 MUST label lowest CV MAE, selected family, and fastest fit model separately and MUST preserve limitations and safe next actions.
- **FR-008**: Page 05 MUST display tuning context status/results, matched Full/Top-2 fold evidence, holdout metrics, residual/absolute-error evidence, encoded and permutation importance methods, subgroup support flags, q90 uncertainty evidence, scientific outcome, operational assessment, and final Full/Top-2 conclusions.
- **FR-009**: Page 05 MUST visibly distinguish `EVALUATION_HOLDOUT` from CV and historical supplemental test evidence, and MUST state that `INFERENCE_RESERVE` has no metrics/predictions.
- **FR-010**: The UI MUST derive display tables and conclusions only from files listed in the selected validated manifest and MUST reconcile run ID/source references before display.
- **FR-011**: Every displayed table MUST use reader-facing labels, explicit units/precision, hidden technical columns where appropriate, bounded rows, and downloadable authoritative evidence for full detail.
- **FR-012**: Missing optional sections MUST show a local unavailable reason and MUST NOT hide other valid training-validation sections.
- **FR-013**: If no complete compatible pack exists, both pages MUST show a collapsed `Training validation unavailable` expander with a safe explanation, read-only inspection command, and no fallback values.
- **FR-014**: The UI MUST NOT use fixture values, existing `ui_evidence` values, historical primary-pipeline logs, old output namespaces, or hardcoded numbers as training-validation fallback evidence.
- **FR-015**: Streamlit code MUST NOT call fit, tuning, training-validation `run`, pipeline generation, joblib load, prediction, benchmark, publication, or holdout-access functions.
- **FR-016**: Existing Page 04/05 charts, supplemental evidence, downloads, conclusions, and page routing MUST remain unchanged outside the new expanders and independent unavailable handling.
- **FR-017**: Page 06, preprocessing, source data, model bundles, primary pipeline outputs, deployment pointers, and current supplemental `ui_evidence/current.json` MUST remain unchanged.
- **FR-018**: The integration MUST reuse existing complete training-validation artifacts; because no producer/schema contract changes are planned, it MUST NOT retrain or regenerate evidence solely for UI integration.
- **FR-019**: User-facing errors MUST be bounded English messages and MUST NOT render exception traces, credentials, absolute external paths, raw records, or untrusted artifact text as executable instructions.
- **FR-020**: Documentation MUST explain the two independent evidence sources on Pages 04–05, latest-pack selection, unavailable behavior, offline generation/check commands, scientific labels, and non-activation of models.
- **FR-021**: Page 04 MUST include a detailed evidence-derived training transcript that answers “thể hiện cái cách chia fold như thế nào bao nhiêu dòng” by showing, for every outer fold, the parent population, train/validation periods and exact row counts, later-unused rows, added history, holdout/reserve exclusions, overlap/chronology checks, and linked source evidence.
- **FR-022**: The Page 04 transcript MUST show all 25 candidate fold evaluations (five declared models × five outer folds) with model/configuration/fold identity, train and validation metrics, fit/predict timing, row counts, status, and evidence reference, followed by one scoped conclusion for each candidate.
- **FR-023**: Page 05 MUST include a detailed method-ordered transcript for all six tuning contexts, inner folds, stage/trial settings/results/reuse/skip state, matched Full/Top-2 fold evaluations, final fitting declaration, one-time holdout evaluation, explainability/subgroup/uncertainty steps, and final conclusions.
- **FR-024**: The UI MUST distinguish raw execution logs (`training.log`, `events.jsonl`) from an `evidence-derived transcript` reconstructed without metric recalculation from authoritative CSV/JSON rows. Every transcript row MUST include source file/reference and unavailable fields MUST remain explicit null/reason values.
- **FR-025**: Pages 04–05 MUST provide downloads for the complete original `training.log`, `events.jsonl`, `report.md`, `agent_summary.json`, and `model_conclusions.json`, plus an in-memory generated complete transcript CSV. Preview rows MUST be bounded, but downloads MUST not silently truncate.
- **FR-026**: Pages 04 and 05 MUST add a compact always-visible `Training & validation evidence` overview before the collapsed details without replacing or modifying existing charts, tabs, conclusions, or supplemental evidence. With a valid pack it shows verified run identity, evidence counts, and trust boundaries; without one it describes supported capability while explicitly denying that a completed or trusted active-workspace run exists.
- **FR-027**: The current workspace MUST show an additive historical Branch B training report sourced only from the validated active supplemental pack and newest complete source-compatible primary-pipeline audit; this report MUST remain independent from strict `training-validation/v1` availability.
- **FR-028**: Page 04 MUST show one 5W1H record for each of Dummy Median, Linear Regression, Ridge Regression, Random Forest, and Gradient Boosting, including run/model identity, regression target/features, fold periods/scope, evidence-backed role/decision, implemented evaluation method, metrics, limitations, and source references.
- **FR-029**: Page 05 MUST show 5W1H records for the selected family and Full/Top-2 final variants, inherited tuning status/settings, historical locked-test scope, explainability/uncertainty evidence, non-promotion limitations, and source references.
- **FR-030**: 5W1H `Why` text MUST describe declared model role and evidence-backed decision; it MUST NOT invent causal success/failure explanations, untouched-test claims, or unsupported overfit diagnoses.
- **FR-031**: Pages 04–05 MUST show bounded model-relevant audit-event previews and offer the complete original JSONL/manifest/relevant exports with verified checksums. Derived report rows MUST be labelled derived and distinguishable from raw events.
- **FR-032**: The UI MUST identify that the common prepared feature base feeds both branches and Branch B trains salary models after upstream preparation; Branch A cluster outputs MUST NOT silently become Branch B model features.
- **FR-033**: Existing Page 04/05 layout, charts, tabs, KPIs, supplemental conclusions, Page 06, model artifacts, and training logic MUST remain unchanged; additions use native containers/expanders and the visual-first/high-density conventions in `docs/spec-imporve-ui.md`.
- **FR-034**: Because complete compatible historical audit and supplemental evidence already exist and no producer contract is changed, implementation MUST reuse them and MUST NOT rerun the root pipeline or regenerate model artifacts solely to populate the report.

### Constitutional Requirements

- **Data boundary**: This feature reads validated summaries only. It does not access source rows directly, train models, score holdout rows, or access inference-reserve features/targets. Displayed CV, holdout, and reserve roles remain distinct.
- **Artifact contract**: `training-validation/v1` remains produced solely by the offline CLI. The new consumer validates immutable final packs and bundle metadata without model loading. Existing packs are reused; producer output is unaffected.
- **Streamlit boundary**: Pages are presentation-only consumers. No UI control starts inspection, training, tuning, scoring, benchmarking, or publication.
- **User input validation**: There are no new user-entered paths or run IDs. Workspace root comes from established routing. All discovered paths/IDs/timestamps/manifests/tables are validated and bounded before display.
- **Scientific interpretation**: CV family selection, one-time holdout results, runtime evidence, operational readiness, descriptive q90, and existing historical supplemental evidence are labelled separately. No causal, fairness, deployment, calibration, or future-performance claim is permitted.
- **Documentation impact**: Update the model UI guide and README link/description; document new loader/renderers and artifact/source boundaries. Do not alter training methodology documentation except to link the display.
- **Verification evidence**: Failing-first pure loader/view tests, transcript source/parity/order tests, corruption/path tests, no-fit/load spies, AppTest through `streamlit.py`, full regression tests, Ruff/diff checks, protected-hash comparison, and browser/human visual review when available.
- **Graphify/Karpathy review**: Graphify identified `page04_model_comparison.py`, `page05_best_model.py`, `model_evidence.py`, `ui_evidence_io.py`, and `training_evidence_io.validate_complete_pack` as the narrow integration path. The simplest approach is one new read-only presentation module plus small page calls; no pipeline merge, pointer, service, or component framework.

### Key Entities

- **TrainingValidationPackDescriptor**: Selected run ID, generated timestamp, manifest identity, availability status, source paths, and bounded invalid-candidate reasons.
- **TrainingValidationView**: Validated JSON records and bounded DataFrame projections for Page 04 and Page 05, with section availability and source references.
- **TrainingTranscriptRow**: A method-ordered, evidence-derived record with stage/step/model/fold/search/trial/partition identities, exact counts/periods, parameters, metrics, status, conclusion and authoritative source reference. It is not represented as a raw runtime event.
- **UnavailableState**: Stable reason code, safe English explanation, inspection command, limitations, and next action; contains no fallback metrics.

## Success Criteria

### Measurable Outcomes

- **SC-001**: In AppTest, Pages 04 and 05 each expose the expected training-validation expander and render without exceptions for valid, missing, corrupt, and partial-pack fixtures.
- **SC-002**: For a valid fixture, every displayed fold/model/final metric sampled by tests exactly matches the authoritative CSV/JSON value before display rounding.
- **SC-003**: All five outer folds, 18 inner folds, five candidate conclusions, and two final-variant conclusions from the fixture are discoverable from the corresponding page UI or its authoritative download.
- **SC-004**: No-pack and corrupt-pack tests show an explicit unavailable state and zero substituted training-validation metrics.
- **SC-005**: Spies prove zero fit, tune, predict, benchmark, publication, pipeline, joblib-deserialization, and holdout-access calls during loader tests and Page 04/05 AppTest reruns.
- **SC-006**: Existing Page 04/05 behavior tests and the full regression suite continue to pass without changing Page 06 behavior or protected artifact hashes.
- **SC-007**: A reader can identify the exact run, fold 3 train/validation periods and row counts, selected-vs-fastest distinction, holdout scope, reserve exclusion, every model decision, and unavailable next action within five minutes.
- **SC-008**: Fixture tests reconcile 25 candidate fold transcript rows, five outer and 18 inner fold descriptors, all tuning trial/context rows, ten matched variant-fold rows, seven holdout-role summaries, five candidate conclusions and two final conclusions to their exact authoritative source rows.
- **SC-009**: The UI visibly labels raw logs versus evidence-derived transcript and offers byte-identical downloads for five original log/report/summary files plus a deterministic, complete, non-truncated transcript CSV.
- **SC-010**: A Vietnamese/English reader can answer how folds are split and how many rows each contains from Page 04 without opening an offline file, and can trace every displayed answer to a source artifact.
- **SC-011**: Real-entrypoint AppTest finds the always-visible overview on both pages in valid and unavailable states while all pre-existing page visualizations and source-local failure behavior remain unchanged.
- **SC-012**: In the current workspace, AppTest discovers exactly five Page 04 candidate 5W1H records and the required Page 05 selected/final records, each with Who/What/When/Where/Why/How, evidence scope, limitation, and source reference.
- **SC-013**: Every sampled 5W1H metric/configuration/fold value reconciles to the validated supplemental pack or compatible audit export, and no strict training-validation field is populated from historical evidence.
- **SC-014**: No-operation/write spies and protected hashes prove the historical report triggers no fit, prediction, pipeline run, artifact regeneration, pointer change, or model deserialization.

## Assumptions

- The existing `training-validation/v1` producer and `validate_complete_pack()` remain authoritative and unchanged by this feature.
- No current real pack exists; the initial real-workspace UI therefore shows the confirmed unavailable state.
- A future approved offline run creates immutable final namespaces but no `current.json`; latest valid UTC manifest time is the agreed selection rule.
- Existing supplemental `ui_evidence` remains the primary content already shown on Pages 04–05; training-validation evidence is an additional clearly separated experiment, not a replacement or blended metric source.
- Expanders are collapsed by default. Reading and validation may happen during page rendering, but no expensive model operation is permitted.
- Existing `training.log`/`events.jsonl` may be concise. Detailed UI steps are therefore an explicitly labelled deterministic transcript projection over authoritative evidence tables, not fabricated runtime telemetry and not a producer rewrite.
- Browser pixel acceptance may remain pending if browser tooling is unavailable; AppTest and pure rendering tests are still mandatory.
