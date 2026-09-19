# Technical Research and Decisions

**Date**: 2026-09-19  
**Status**: Baseline and approved readability remediation implemented; research decisions retained as the design record.  
**Evidence**: [Initial and follow-up inspection](evidence-review.md), [assessment](../../.specify/assessments/improve-model-ui/research.md), [spec](spec.md).

## R1 — Add an offline supplemental command, not code in the monolith

**Decision**: A new `src/ai_job_market/ui_evidence.py` module provides `python -m ai_job_market.ui_evidence`. Delegate fitting/scoring to `ui_evidence_training.py`, artifact contracts to `ui_evidence_io.py`, and observed-role policy to `scenario_policy.py`. `salary_inference.py` handles validated serving and exports, without importing the offline orchestrator. Reuse existing `candidate_models`, `temporal_cv_splits`, `make_model_pipeline`, `regression_metrics` and `extract_encoded_importance` as functions; do not edit `core.py` or monkeypatch its evaluator.

**Rationale**: User explicitly prohibits adding evidence-generation logic to the god module. Existing helpers support arbitrary feature subsets, so no primary-trainer refactor is required.

**Alternatives rejected**: App-triggered fitting violates the offline boundary. Adding another large function to `core.py` violates the user decision. Extracting/rearchitecting all legacy training would expand scope unnecessarily.

## R2 — Supplemental namespace and per-component reuse

**Decision**: Store report files under `outputs/ui_evidence/<evidence_id>/`, generated Top-2 model under `artifacts/ui_evidence/<evidence_id>/`, and an atomically replaced `outputs/ui_evidence/current.json` pointer. Keep original output packs and bundles byte-for-byte unchanged. A manifest binds source file hashes, ordered data identities, core/new producer code, configuration, dependency lock/runtime versions, schemas, seeds and estimator parameters. Reuse a valid component only when all its dependency identities match; changed policy/export wrappers must not trigger unnecessary estimator fitting. New manifests may reference immutable previous component files with matching hashes.

**Rationale**: The legacy Train_* fields do not match the inspected producer; a reproducible supplemental contract is safer than blessing them or silently overwriting them. A valid unchanged repeated invocation must make zero fit calls.

**Alternatives rejected**: Re-running `pipeline.py` also trains unrelated segmentation. Updating old CSVs in place loses provenance and risks Pages 07/08. One fingerprint over all UI source would cause needless fits for presentation changes.

## R3 — Reconstruct train and validation metrics together

**Decision**: Freeze current candidate factory configurations and existing fold memberships, then use a small new evaluator to fit each candidate once per DEV fold and predict both that fold's training and validation rows. Emit MAE/RMSE/R²/MedAE for both, actual dimensions, timings and immutable membership references. Capture RF encoded importance during those same fits; compute raw-family drift from per-fold sums before mean/SD, not sums of feature SDs. Regenerate paired scores rather than combine new train values with unverifiable old validation numbers.

**Rationale**: `core.evaluate_model_cv` exposes validation scores only. New diagnostic fits are now within approved scope, but they must not be confused with a reproduction of another historical configuration. Current candidate definitions differ from source-document chips.

**Alternatives rejected**: Training metrics on the final all-DEV model are not fold-training metrics. Hardcoding document values lacks evidence. Reusing previously fitted fold models is not possible when they were not saved.

## R4 — Frozen Top-2 experiment, no new search or primary-model promotion

**Decision**: Require the compatible selected full model to be Random Forest for the requested RF Top-2 mode. Read/verify its actual applied estimator parameters against metadata; clone those settings (seed included) for the two inputs `job_category`, `years_of_experience`. Evaluate frozen full and Top-2 settings on identical existing DEV folds; fit Top-2 once on all DEV; load rather than refit the saved full model. Never select settings by historical test results. If full family/configuration evidence is incompatible, stop the affected component rather than substitute a factory default. Retain Page 04's candidate ranking separately from the saved selected-model identity; disagreement is a visible warning, not automatic replacement.

**Rationale**: This provides a fair fixed-configuration feature comparison and independent Top-2 metrics without introducing a new tuner. It does not claim Top-2 is the optimally tuned model.

**Alternatives rejected**: Hardcoded $34,101/R² 0.815 is unsupported. Retuning to recover illustrative scores is test chasing. Applying the 13-feature bundle to two columns is invalid. Replacing the selected full model would change primary-training behavior.

## R5 — Historical test evidence, not a new pristine holdout

**Decision**: Preserve all 298 currently scored test rows (actual count remains data-driven). Before reading test targets, persist the frozen evaluation declaration with configurations, feature sets, fold identities and the no-selection rule. Score the frozen five candidates on the historical test after final DEV fits; score Top-2 and verify saved full predictions/metrics. Produce both raw permutation and encoded importance for full/Top-2 (reuse full evidence only when compatible). These are retrospective diagnostics; no configuration is promoted from test performance.

Choose up to three benchmark examples from the historical test using DEV-supported pairs and title bounds, ascending stable record identity, with no target/error/salary-tier selection. Record `included_in_benchmark=true`, `historically_exposed=true`, `pristine=false`. Preserve the full test denominator. Real pristine data acquisition and 295+3 repartitioning are out of scope.

**Rationale**: The available data cannot become unseen by retraining. Candidate-level test metrics improve reporting coverage but do not repair historical selection exposure. The user approved offline generation, not fabrication of new data.

**Alternatives rejected**: Cherry-picking three accurate examples hides failures. Holding out previously scored records now cannot establish pristine status. Excluding three rows while reusing 298-row metrics mislabels the population.

## R6 — Observed DEV pairs and title-level fractional bounds

**Decision**: Derive category/title pairs and title-level min/max/median/support counts from DEV input fields only. Intersect with verified metadata enums; reject incompatible enum metadata, do not silently invent or normalize labels. A title may appear under multiple categories. Preserve fractional medians (e.g. 7.5), using finite numeric widgets and an appropriate fractional step rather than integer truncation. Label the badge `Within observed DEV bounds`, not `Profile in-distribution`.

**Rationale**: DEV-only policy avoids test-informed serving rules. Observed pairs implement the user's choice but cannot certify logical business taxonomy. Source data is inconsistent.

**Alternatives rejected**: Full cleaned-data bounds incorporate test information. A curated title-name heuristic would contradict the user's choice. Rounding the median changes the specified default.

## R7 — Strict default with bounded experience exceptions

**Decision**: Builder, benchmark and growth generation use one policy function. Manual queue rows never bypass bounds. Default quick-load records are strict-valid. Only an immutable known benchmark whose experience is within global exploratory 0–15 but outside title bounds, or an explicitly requested curve extension within 0–15, may be acknowledged as an exception. All enum/pair/type/model checks remain mandatory. No free-text reason: use fixed reason `experience_outside_observed_dev_range`, preserve acknowledged evidence/policy identity and reset acknowledgement when either changes. Exceptions use warning badges, dashed curve segments and propagated export flags; empirical q90 is shown only as a historical reference with unvalidated extrapolation coverage.

**Rationale**: Implements 'exceptions allowed, uniform bounds common' without turning acknowledgement into a general validation escape hatch. Current strict-valid benchmark candidates are plentiful, so the normal three examples need no exception.

**Alternatives rejected**: A global 'disable validation' switch is unsafe. Silent clipping changes audit records. Automatically extending every curve to 0–15 makes exceptions the normal path.

## R8 — Preserve UI routing; test calculations outside Streamlit

**Decision**: Keep `render(st, root, role)` and existing `streamlit.py` routing. Use native tabs/containers/expanders and existing Plotly, which already supports the requested dual-axis/whisker charts. Use `width="stretch"` in touched rendering code; no CSS expansion, custom component, navigation rewrite or dependency upgrade. A small `src/pages/model_evidence.py` binds cached loaders to the active workspace/evidence identity and provides pure figure/table builders.

Reactive category/title/experience controls remain outside a batching form so constraints update on each selection. Callbacks reset dependent widget state before rendering. Queue/result state includes workspace, evidence ID, policy hash and revision; a change clears stale results and revalidates/clears the queue.

**Rationale**: Existing Plotly and routing are adequate. AppTest can exercise entrypoint navigation and session transitions; plain pytest tests pure helpers and figure data. Browser checks are still required for clipping/legibility, not replaced by AppTest.

**Alternatives rejected**: Switching plotting stacks for a preference adds churn without value. Putting all dependent widgets inside the existing form prevents immediate role-bound updates. Caching only by filename leaks stale evidence across workspaces.

## R9 — Explicit signed variance and source-schema export

**Decision**: Publish a new versioned prediction-audit CSV with `signed_variance_pct`; do not silently change old `error_pct` from absolute to signed. Retain a separate original-25-column scenario export using `SOURCE_COLUMNS` order, mapping `job_category` to the source's `AI Engineering` column. Leave unknown original attributes and salary fields empty for manual scenarios; known benchmark source values may be retained with audit identity in the separate audit CSV. Synthetic scenario identifiers must be visibly scenario IDs, not purported real job IDs. Do not invent current posting dates.

**Rationale**: The current exporter drops category because it writes `job_category` then selects `SOURCE_COLUMNS`, which contains `AI Engineering`. New filenames/schema versions make the export migration intentional.

**Alternatives rejected**: Adding prediction columns to the original schema breaks 'original schema'. Filling observed salary with predicted salary confuses forecasts with source facts.

## R10 — Quantified diagnostic indicators rather than unsupported fit labels

**Decision**: Report actual train/validation gap, ratio and baseline comparison. Label statuses deterministically as `Reference baseline`, `Below baseline`, `At/above baseline`, or `Insufficient evidence`; a positive train-validation gap is descriptive, not a standalone overfit classifier. Preserve requested fit-spectrum educational context in a collapsed explanation: good/under/overfit are hypotheses requiring more evidence. Do not label a numeric threshold as proof of memorization, near-singularity or causal salary economics.

**Rationale**: Constitution V requires evidence-backed interpretation. No accepted three-way classification rule or statistical test was provided.

**Alternatives rejected**: Inventing gap thresholds to reproduce 'Good fit'/'Underfit' labels would create an unsupported model assessment contract.

## R11 — Scope scientific and UI governance exceptions honestly

**Decision**: Preserve inherited tuning history with an explicit `inherited_non_nested_dev_search` label and original R² ranking; new work has no hyperparameter search, family reselection or test-driven promotion. Capture known historical VI.3/VI.4 discrepancies and current global upload-training boundary issue in the plan's Complexity Tracking. All new producer work runs only through the CLI; Pages 04–06 never call it. Existing source-text UI tests must become behavior/contract assertions where tab/filter counts change, not simply deleted.

**Rationale**: Changing old experiments or global upload workflows is not required to build this feature and risks unrelated consumers. Bounded observation of inherited evidence is not an assertion of unconditional constitutional compliance.

**Alternatives rejected**: A fresh nested tuning platform, primary-model replacement or global upload rewrite would be a separate substantial feature. Ignoring the inherited issues would be dishonest.

## R12 — Validation and generation order

**Decision**: Build failing-first tests with small fixed in-memory fixtures and estimator spies; exercise producer tests in temporary workspaces. After implementation approval and successful tests, run the new CLI once for missing/invalidated real supplemental artifacts, then `--check` and an unchanged rerun to prove reuse. Do not run the full root pipeline for a UI demo. Page tests/consumer assertions use the resulting saved pack with fit/tune calls forbidden. Record browser evidence at 1280/1440 widths and warm/cold timing separately.

**Rationale**: Verification must cover both the changed producer and its consumers without repeatedly regenerating unrelated training outputs. Current outputs can remain reference evidence only within their verified identities.

## R13 — Treat legibility as a data contract, not a styling preference

**Decision**: Every figure receives an explicit readability profile: intended question, priority, minimum useful width/height, margins, label policy, axis formatting and dense-data fallback. Shared rendering may apply harmless defaults, but it must not overwrite figure-specific margins. Use Plotly `automargin`, deliberate height, wrapped/short display labels, `cliponaxis=False` where appropriate and table/hover detail as the fallback when direct labels would collide.

**Evidence**: Runtime AppTest inspection of the real evidence pack found that all Page 04–06 figures were forced through `show_plot` to the same `l=20, r=20, t=55, b=25` margins. Page 04's drift plot places 12 labels up to 19 characters inside a 680-pixel P3 container. Page 05's four tuning figures place 8–12 direct labels per chart while each chart occupies half a row; encoded importance shows 15 horizontal categories with only a 20-pixel left margin. These conditions make clipping/overlap plausible and match the stakeholder report that numbers cannot be read. See `ui-readability-audit.md`.

**Rationale**: Width priority alone does not guarantee readable marks. A compact chart must have fewer labels or a layout suited to compact width.

**Alternatives rejected**: Making every chart full width repeats the original hierarchy problem. Shrinking fonts hides information. Adding CSS cannot fix Plotly's internal label geometry reliably.

## R14 — Progressive disclosure for a third-party report reader

**Decision**: Pages use a three-layer information model: (1) a plain-language page brief and KPI takeaway, (2) one primary decision chart and a derived conclusion, and (3) supporting diagnostics/tables in tabs or expanders. Add a compact `How to read these metrics` expander defining MAE, MedAE, RMSE, R², CV, residual and q90 in business language. Every section states the question it answers before showing evidence.

**Evidence**: Current pages lead with technical captions such as `frozen-configuration CV` and immediately present multiple charts. Conclusions are mostly limitations (`not causal`, `not proof`) and do not state the observed result, practical meaning or next decision. Local Streamlit guidance recommends clear labels, progressive containers, captions for metadata and expanders for optional detail (`references/design.md`, `layouts.md`, `data-display.md`, `dashboards.md`).

**Rationale**: A third party should not need prior knowledge of the pipeline or report structure to identify the answer and its limits.

**Alternatives rejected**: Long prose before every chart makes scanning worse. Tooltips alone are undiscoverable and inaccessible for a report reader. Static conclusions risk becoming false when evidence changes.

## R15 — Derived conclusions use a fixed evidence/meaning/limit/action grammar

**Decision**: Add pure conclusion builders that accept validated evidence and return four concise fields: `Finding`, `Why it matters`, `Limit`, and optional `Decision/use`. Conclusions must include the compared values/population and be recalculated when evidence changes. If required evidence is missing, state `Conclusion unavailable` with the missing source rather than emitting generic prose.

Page-specific minimum conclusions:

- Page 04: winning candidate and runner-up MAE gap; improvement versus Dummy; fold range/worst period; family-ranking scope versus saved-model scope.
- Page 05: historical-test versus final-config CV deltas; q90/coverage/tail concentration; full versus Top-2 comparison; tuning and historical-exposure limitations.
- Page 06: active model/domain; batch prediction range; uncertainty basis; known-actual variance when available; extrapolation warning when used.

**Rationale**: This supplies the missing report narrative while preserving scientific honesty and preventing unsupported fit or causal claims.

**Alternatives rejected**: Handwritten model verdicts become stale. A single generic warning does not answer the reader's question. LLM-generated runtime commentary would add nondeterminism, dependencies and unsupported interpretation.

## R16 — Label density rules override literal label-every-point behavior

**Decision**: Keep direct labels on primary values, but reduce redundant labels when a chart is dense. For paired bar/line charts, label all bars and only key line points (latest, minimum/maximum or selected optimum) when width is constrained; all exact values remain in hover and the evidence table. For scatter plots, never render hundreds of point text labels. For 10-scenario prediction pages, label prediction and known actual, while interval endpoints move to compact hover/table or a separate annotation only when collision-free. Horizontal importance charts reserve enough left margin for feature names and show at most the evidence-backed Top N selected by the view contract.

**Rationale**: The stakeholder's core need is visible numbers, not the maximum number of text glyphs. Selective direct labels increase the number that can actually be read.

**Alternatives rejected**: Labelling every point in small charts caused overlap. Removing all labels violates visual-first review. Truncating evidence from the table/export would reduce auditability.

## R17 — Use a two-tier table system for visible emphasis and detailed exploration

**Decision**: Page 04–05 use compact `st.table` decision summaries where bold/italic semantics matter and interactive `st.dataframe` detail tables where width, sorting or complete evidence matters. Compact tables use Markdown because the installed Streamlit 1.64 `st.table` contract renders GitHub-flavored Markdown; interactive dataframes do not render inline Markdown and therefore receive typed `column_config` formatting instead. A focused `model_training_presentation.py` helper applies sentence-case headers, display units and the legend `**Bold = governing result** · *Italic = reference/context*` without changing the shared figure module fingerprint used by the offline evidence producer.

**Emphasis rules**: On Page 04, the winner is the finite minimum mean validation MAE using unrounded values; Dummy is an italic reference; ties share the winner emphasis and rank. On Page 05 tuning, the per-stage winner is the finite maximum CV R² because that is the inherited procedure; the saved applied value is bold in the configuration summary, while a stage winner that differs remains labelled as sensitivity evidence rather than applied configuration. The scorecard uses bold for the consequential historical-test column and italic for DEV-CV context, not to suggest test results selected the model.

**Rationale**: This satisfies the requested typography without sending raw `**Markdown**` into `st.dataframe`, without forcing a 12-column executive table into a noninteractive static layout, and without introducing CSS. Formatting remains a presentation derivation; exports retain unmodified machine-readable evidence.

**Alternatives rejected**: Applying Markdown strings to `st.dataframe` displays literal markup. Styling every cell via Pandas Styler would make emphasis harder to test and conflicts with the local guidance to use `column_config` for dataframe value formatting. Converting all tables to `st.table` would make wide/detail evidence less usable.

## R18 — Describe the implemented temporal splitter, not generic K-fold

**Decision**: Add a prominent collapsed Page 04 guide before the tabs. Use the stakeholder's “K-fold temporal validation” phrase only with an immediate correction: the implementation is a temporal sliding-window scheme, not randomized K-fold and not an expanding window. Chronologically sorted DEV rows are partitioned into contiguous row blocks; each fold trains on one block and validates on the next, with the final validation block receiving the remainder. Every candidate pipeline and its preprocessing are refit on the training block only. The guide derives fold count, row counts, validation periods and encoded dimensions from the active supplemental tables and offers fold metrics plus membership downloads.

**Rationale**: `temporal_cv_splits` explicitly uses adjacent one-block train/validation windows. Calling this generic K-fold would hide an important scientific limitation. Adjacent blocks can share a month label while retaining disjoint row identities, so the guide must distinguish calendar overlap from row overlap.

**Alternatives rejected**: A textbook K-fold explanation would misdescribe the code. Expanding-window wording would be false. Reconstructing historical fold details from prose or audit logs would be weaker than the active pack's current membership evidence.

## R19 — Explain inherited tuning as anchor search plus coordinate sensitivity

**Decision**: Expand Page 05's tuning methodology into `What`, `How`, `Why`, `Selection rule`, `Applied configuration` and `Limitations`. The actual workflow evaluates four initial RF configurations, ranks trials by descending five-fold temporal CV R², then evaluates 6 `n_estimators`, 6 `max_depth`, 4 `min_samples_leaf` and 6 `max_features` values sequentially. Later stages carry prior values and the implementation fixes `max_depth=20` for subsequent sweeps. The final estimator is created from the first row of the R²-sorted initial tuning table; later sweeps are supporting sensitivity evidence and do not silently redefine selection. Where current evidence shows equality, say the saved values match the recorded stage optima; never infer causality from the inherited `tuning_rationale` strings.

**Rationale**: This is the most important distinction for answering how/what/why accurately. Page 04 ranks families by MAE, whereas tuning ranks RF settings by R². The historical test did not choose these settings, but the DEV reuse is non-nested and therefore not an independent estimate of tuning gain.

**Alternatives rejected**: Describing generic grid search versus random search adds irrelevant textbook content. Saying all four sweep winners built the final model contradicts `final_model_from_selection`. Showing legacy rationale text would repeat unsupported overfit/causal claims.

## R20 — Expose optional audit traces only through a compatibility gate

**Decision**: The active supplemental pack remains the sole source of displayed Page 04–05 calculations. Add a read-only loader in `model_training_presentation.py` for the fixed `outputs/08_full_pipeline/logs/` directory that considers only complete manifests (`training_status=completed`, complete log/console/coverage), requires the current raw source SHA-256, verifies every offered export checksum and refuses escaping/symlinked paths. The UI shows audit/pipeline run IDs and labels the trace as the primary-pipeline historical audit, distinct from current supplemental paired metrics. Relevant fold/tuning exports, manifest and complete JSONL may be downloaded; they never overwrite active-pack tables. Missing or incompatible logs produce a local unavailable reason and never invoke training.

**Rationale**: Existing logs retain complete trial/fold evidence and are useful for audit/export, but they belong to a separate run identity. A compatibility gate prevents silent mixed-run science while allowing the user to inspect the available provenance.

**Alternatives rejected**: Copying logs into the current evidence pack would mutate/reversion artifacts for a presentation-only feature. Selecting the newest filename without validating completeness/fingerprint is unsafe. Parsing logs into new scientific metrics duplicates authoritative CSV evidence and risks semantic drift.

**Unresolved technical decisions**: Browser screenshot/DOM inspection is still required to calibrate exact heights and label thresholds. Chrome DevTools MCP is not configured in this harness, so AppTest protobuf inspection supports diagnosis but does not count as visual acceptance. The audit-log compatibility rule is specified for implementation tests; if the current complete runs fail it, the UI must show logs unavailable rather than weaken the gate.
