# Research: Readable Training Logs

**Date**: 2026-09-18  
**Method**: Existing Graphify query, direct source inspection, test-contract inspection and installed-package metadata. No training or test suite executed. No research-agent tool is available; investigation was performed directly, with independent file reads in parallel. Sources below refer to the working tree, including inherited uncommitted changes, not just HEAD.

## Evidence snapshot

HEAD: `4a02508dc88f6c9abe52360e0b12a89e5975d4d7`; branch: `004-readable-training-logs`.

| File | SHA-256 at planning |
|---|---|
| `pipeline.py` | `1eaf6475927312ef942c22eb37fe3cb2d2ed7eabe7ee7cc08467cf2c2374b944` |
| `src/ai_job_market/core.py` | `92df32ec7c630a9af9a2a890ff06afb1fca8ce0f28b305c9f6f705c6f846f885` |
| `src/ai_job_market/training_audit.py` | `9793ea3bdaa1371ac94eb62cf4594c29ed0aa467359540b1684b300c58cb8680` |
| `config/project.yaml` | `da5d23253b328f2594d345509c48931f5993a3d503588c10690e95f323f116a3` |
| `requirements.txt` | `8e2f2581ea4d4ab13dc033c51c7f0654a9fee1e694c89762813464562934da4b` |

Graph query: `graphify query 'training audit pipeline stage fold tuning comparison diagnostic console coverage' --budget 2200`. It found 122 nodes and displayed 69; the truncated graph is orientation, not a coverage proof. Direct inspection verified the core training/audit chain, all of `_run_pipeline_impl`, `run_segmentation`, and CLI lifecycle. `graphify reflect --if-stale` found zero prior memories. Re-export facades in `src/training/temporal_cv.py`, `src/training/tuning.py` and `src/pipeline/model_training_comparison.py` contain no alternative trainer.

## R1 — Extend the current audit, do not replace the trainer

**Evidence**: `training_audit.py:74–272` already owns a context-local session and strict JSON file; `core.py:3817–3832` wraps the existing workflow. The current event sink prints JSON, then writes the file in the same try block. A broken terminal can therefore prevent file writes. Initialization and close failures are not covered by the event write fallback.

**Decision**: Retain the session and file convention; introduce a small human renderer and independently guarded file/terminal writes. Persist first and filter presentation only. Keep warning/error severity separate from information depth. Use explicit event schemas rather than dumping arbitrary values.

**Rationale**: Reuses current instrumentation, avoids a new trainer or event infrastructure, and makes the two outputs testable independently.

**Alternatives considered**: Rich (not installed, not required for readable headings/tables); replacement with root logging configuration (risk of unrelated output changes); standard-library logging handlers (valid alternative, but replacing the existing functioning session adds migration and exception-routing complexity with no needed feature). Keep explicit session-owned sinks and plain-text rendering; use existing terminal color capability only as optional decoration. No dependency changes.

## R2 — Stage completion must come from execution, not file existence

**Evidence**: `pipeline.py:325–413` defines 12 display stages; `artifact_written` advances on save suffixes, while `finalize_unobserved` at 552–575 marks every remaining stage PASS. Segmentation's marker is written before its bundle; marker completion is not full stage completion. Source wrappers only observe CSV/JSON writes.

**Decision**: Emit explicit stage start/end events in `core.py` at actual boundaries listed in [coverage.md](coverage.md). Make `TerminalProgress` consume those events for timings and heartbeats; remove marker hooks from the main execution path. Preserve timing output filenames/columns but report unobserved stages as UNKNOWN, never PASS. Cancellation is distinct from failure. A returned pipeline may be scientifically successful with incomplete observation; display both statuses separately.

**Alternatives considered**: Keep marker hooks and adjust filenames (still misses failures after markers and unobserved operations); infer success from stale output files (not reliable); refactor pipeline into stage classes (unnecessary scientific risk).

## R3 — Actual data counts and split provenance

**Evidence**: `basic_clean` at `core.py:432–459` already computes invalid-category and duplicate counts in order, then drops job_id and resets index. `temporal_split` at 849–868 selects preferred latest period or latest available and resets each partition index. `temporal_cv_splits` at 949–987 uses stable chronological row blocks, normally 200, not expanding-window CV. The last validation slice absorbs the remainder. `_temporal_fold_audit_payload` hashes only `0..n-1` for its current row digest, which cannot distinguish different same-sized datasets.

**Decision**: Log the existing audit dictionary's safe numerical fields, actual feature lists and produced shapes. Add a content-and-order fingerprint of the in-memory DEV frame, together with source fingerprint, derivation and run identity; add this as a new field rather than relabeling the old digest. Use the actual split arrays already produced; never call the splitter a second time for logging. Export zero-based DEV-relative membership with explicit dataset identity and no raw records. Do not equate the old positional digest with a content hash.

**Alternatives considered**: Recompute cleaned inputs/splits (could diverge and violates no-extra-split rule); use index labels alone (duplicate/reset indices); trust counts as identity (collision for same-sized datasets).

## R4 — There is no empirical train/validation fit classifier today

**Evidence**: `evaluate_model_cv` at `core.py:1054–1173` predicts validation only. It reports four regression metrics and MAE population SD (`ddof=0`), but does not emit the returned candidate summary. Repository search found narrative overfitting statements in `core.py:2963` and presentation text, not paired train/validation diagnostics or a tested classification rule.

**Decision**: Every evaluated candidate gets a fit-assessment entry with `insufficient_evidence`, reason `training_scores_not_computed; diagnostic_rule_not_defined`. Display actual CV metrics and dummy baseline differences as descriptive comparisons only. Do not invent thresholds, add training predictions or reuse UI claims. Support documenting the three possible labels in the contract, but do not build a speculative classification engine in this feature. Log-only mean/population-SD summaries for all four metrics may be derived from already computed fold rows; keep existing output tables unchanged.

**Alternatives considered**: Train–test gap as proof of good fit (different partitions and fitted models); arbitrary R² thresholds (unsupported); extra training predictions (explicitly outside spec). Future classification with additional measurements requires a separate approved scientific scope change.

## R5 — Tuning traceability must describe the actual choices

**Evidence**: `core.py:2761–2997` executes 4 + 6 + 6 + 4 + 6 = 26 RF trials when RF wins. Every tuning table is ranked by descending CV_R2; later depth is fixed to 20. `_run_pipeline_impl` assigns `tuning = s1`, and `final_model_from_selection` uses its first row. Family comparison sorts by MAE. Tuning descriptions contain fixed narrative values; those are not fresh measurements. Tuning is skipped for non-RF winners. `evaluate_model_cv` is reused for ablation, family comparison and tuning, so names alone are not a sufficient parent identity.

**Decision**: Give each evaluation invocation an observation-only ID and stage/trial context. Include full effective model parameters/seed, trial order, fold IDs, selection criterion/direction, selected row, final applied parameters and differences from later-stage recommendations. Default: stage/candidate summaries and per-trial progress; debug: fold/parameter detail. Existing narrative rationales are retained in their old artifacts but labeled legacy narrative if referenced, not rendered as fit evidence.

**Alternatives considered**: Change ranking to MAE or consume s5 (alters science); fit every model on locked test (new held-out exposure); parse trial identity solely from display names (fragile).

## R6 — Complete evidence without retaining every table in memory

**Evidence**: Existing CSV outputs cover cleaning, model comparison, tuning, feature importance, segmentation and predictions; `save_csv` currently emits path and row count only. Some result tables and membership arrays are not persisted as independent CSVs. `.logs` records can already contain very large candidate/membership payloads.

**Decision**: Keep full JSON evidence in the existing `.logs` stream. Add a uniquely run-scoped sibling evidence directory, CSV exports and manifest. Export only safe evidence tables, not copies of raw input/prediction rows. Stream each table on availability, preview at most 20 rows and discard the preview buffer. Snapshot relevant existing numerical tables into the run area when needed to preserve evidence across later overwrites; never copy all output packs. Record source path and fingerprint of snapshots. Do not reread historical outputs to fill a current-run gap.

**Alternatives considered**: Unbounded terminal output (unreadable); truncating files together with console (loss); external database (unnecessary); references to mutable files alone (old runs lose traceability). No CSV replay CLI is necessary; ordinary CSV readers suffice.

## R7 — Coverage beyond salary metrics

**Evidence**: `_run_pipeline_impl:3286–3814` also runs ingestion, quality checks, preprocessing readiness, Branch A segmentation, bundle writes/reload, prediction examples, integrated outputs and final status. `run_segmentation:1864–2708` produces O1/R0–R4, official O1, O2 and selection evidence, including candidate and resampling tables.

**Decision**: Cover every top-level workflow stage and explicit salary training operation. Add Branch A substage boundaries around existing encoder, representation, each evaluation invocation, O2, selection and evidence construction; report the existing returned candidate/resampling tables. Do not instrument every estimator-internal or resample iteration. State this boundary in terminal help/docs and coverage tests: complete observable workflow coverage is not per-tree/per-solver tracing. No changes to segmentation algorithms or helper-module internals.

**Alternatives considered**: Only log B4/B5 (misses preparation and long Branch A work); instrument every helper fit/resample (large unrelated scope with repeated output); claim all functions are traced (false).

## R8 — Compatibility, runtime and verification

**Evidence**: All three audit test modules parse JSON from stdout and must migrate to file assertions. Behavior tests already characterize split/fit/predict counts and RF selection. `.venv/bin/python` is 3.13.12; installed sklearn 1.9.1, pandas 2.3.3, NumPy 2.5.3, pytest 9.1.1. The shell's `python` is 3.14.4. Ruff targets 3.12. `pipeline.py` has no workspace option and writes into its source tree.

**Decision**: Use the existing venv without upgrades. Add keyword-only debug and optional event observer arguments without changing positional callers or return values. Run real CLI validation only from fresh copied projects; run the core with `workspace_root`. Preserve legacy file envelope fields and add optional fields under schema_version 1; terminal JSON removal is an intentional documented interface change. Verify semantic parity, not timestamp/byte equality. Keep tests that enforce underlying science.

**Alternatives considered**: Run root CLI for convenience (overwrites release outputs); change dependency pins during logging work (confounds parity); count previous reported runtime tests as new evidence (invalid).

## R9 — Governance and the user's educational-project constraint

**Evidence**: Constitution VI.2 requires candidate locked-test reporting; VI.3 says nested development tuning; VI.4 requires MAE selection/tie simplicity. Current source has selected-model-only test reporting, repeated use of the same CV folds rather than an outer nested loop, R² tuning and no explicit variance/simplicity tie gate. Existing quality/segmentation summaries also expose whole-dataset descriptive statistics before salary selection; the test is already historically exposed.

**Decision**: No new security workstream, auth system or dependency hardening. Retain existing path checks and exclude secrets as minimal repository obligations. Record inherited scientific discrepancies as an explicit implementation gate, not silent fixes or constitutional approval. Documentation/design/tasks may be prepared; the implementation cannot be called constitution-compliant until maintainers resolve the findings. No exception can authorize additional leakage.

**Alternatives considered**: Ignore constitution because educational (not permitted by repo governance); silently change experiments (violates user scope); stop documenting code gaps (would defeat the requested accurate plan).

## Completion and limitations

All technical design unknowns have concrete decisions. Scientific governance disposition and implementation approval remain explicit gates, not guessed answers. No live performance or parity result is claimed. Source coverage review is bounded to workflow orchestration and training/audit contracts; delegated estimator internals are not exhaustively audited. The requested post-plan evidence challenge is recorded separately in `.specify/assessments/readable-training-logs/research.md`.
