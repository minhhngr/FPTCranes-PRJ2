# Implementation Plan: Readable Training Logs and Evaluation Evidence

**Branch**: `004-readable-training-logs` | **Date**: 2026-09-18 | **Spec**: [spec.md](spec.md)  
**Input**: `specs/002-readable-training-logs/spec.md`  
**Status**: Implementation approved 2026-09-18 with observation-only disposition. Preserve and disclose inherited VI.2/VI.3/VI.4 differences; do not change training logic.

## Summary

Extend the existing audit session with human-readable, four-depth terminal output: levels 1–2 normally, 1–4 with `--debuglog`. Retain complete structured events in `.logs` files in both modes, with run-scoped CSV evidence for large tables. Explain actual source data, feature variants, count derivations, partitions, folds, every salary candidate/trial and final applied parameters. Never infer fit quality from unsupported narratives or manufacture completed stages.

Keep the trainer, splits, selection and artifacts unchanged. The principal changes are observation calls in the existing core, a small presentation module, and CLI progress driven by actual stage events rather than artifact-name heuristics. [coverage.md](coverage.md) enumerates every top-level stage plus explicit salary operations and bounded segmentation substages. [research.md](research.md) records source-backed decisions and limitations.

User authorized automatic plan → post-plan research → tasks generation, not implementation. Educational context means no security expansion: retain existing boundary checks and safe summaries only.

## Technical Context

**Language/Version**: Python; existing `.venv` 3.13.12, Ruff target 3.12. Shell Python 3.14.4 is not the verification interpreter.  
**Primary Dependencies**: Existing pandas 2.3.3, NumPy 2.5.3, scikit-learn 1.9.1, PyYAML/joblib/pytest; standard-library formatting, JSON, CSV, contextvars and threading. Rich is not installed; do not add it or replace the logging stack for this feature.  
**Storage**: Existing `outputs/08_full_pipeline/logs/training-<timestamp>-<uuid>.logs`; additive sibling `<same-stem>/` with `manifest.json` and safe CSV evidence. Existing output packs and serialized model artifacts remain unchanged.  
**Testing**: pytest 9.1.1; contract fixtures, captured streams, failure injection, estimator/split spies, normal/debug CLI subprocesses in copied projects and core runs in isolated workspaces.  
**Target Platform**: Existing Linux CLI; plain UTF-8 output remains readable without color/TTY, including redirected output.  
**Project Type**: Offline ML pipeline with CLI, importable core and read-only reporting consumers.  
**Performance Goals**: Zero additional fit/predict/split calls; maximum 20 data rows per rendered table; no accumulation of all run tables; persist each available table once per unique evidence ID. Measure time/bytes rather than invent a runtime SLA.  
**Constraints**: Observation/export only; no extra locked-test use, feature/search/seed/ranking changes, root logger configuration, UI edits, dependency upgrades or baseline output overwrite. Existing scientific discrepancies are reported, not silently repaired.  
**Scale/Scope**: Twelve current CLI stages; five salary candidates, four readiness ablations, four family ablations, importance fits and up to 26 RF tuning trials, each with the actual effective fold count. Default requested folds is five; do not hardcode row/fold counts. Branch A logs workflow/evaluation-invocation boundaries and existing tables, not individual resample/solver iterations.

## Constitution Check

**Pre-design review**: Observation-only design passes scope, reproducibility, UI and minimal boundary checks. The inherited scientific findings below block an unconditional compliance claim. Research/source inspection is diagnostic and does not execute ML operations.  
**Post-design review**: Same findings remain; design does not introduce extra scientific exposure. **Bounded disposition recorded 2026-09-18**: the user approved observation-only implementation and explicitly prohibited training-logic changes. The inherited findings remain documented non-compliance risks; this feature may expose them but may not repair or waive them.

| Gate | Design / acceptance evidence | State |
|---|---|---|
| Data integrity | Target `annual_salary_usd`; actual feature variants, DEV/locked period and train-only fitting recorded; no new reads/scoring of held-out targets for logging. Preserve actual folds, including shared months, without claiming strict temporal separation. | Feature design pass; inherited exposure caveats require review |
| Reproducibility | Source/config/DEV content fingerprints, versions, seeds, unique run/evaluation/trial/fold IDs, event sequence and immutable run evidence; existing bundles/metric schema unchanged. | Design pass; runtime proof pending |
| Evidence-first verification | Tests first for rendering, depth, event coverage, summaries, failure handling and exports; semantic output/call-count parity against a captured pre-change working-tree baseline. | Planned, not executed |
| UI boundary | No UI code touched and no new training invocation from UI; facades remain re-exports. | Design pass |
| Input validation | `--debuglog` boolean switch; existing data validation preserved; existing workspace containment and exclusive file creation retained; known current-run CSV schema/identity checked if read. No uploads, auth or remote services introduced. | Design pass; no hardening workstream |
| Scientific interpretation | No fit classifier without evidence; label all partition/aggregation/units and empirical interval limitations; no causal/production claims. | Design pass |
| VI.2 candidate test coverage | Only the final selected model currently has test scoring. Filling candidate test cells would change the experiment. | BLOCKED: maintainer disposition |
| VI.3 nested tuning | Current manual tuning reuses DEV CV evaluations without an outer nested evaluation layer. This feature cannot add it. | BLOCKED: maintainer disposition |
| VI.4 selection/tie policy | Family uses MAE, tuning uses R²; no explicit fold-variance simplicity tie gate. Preserve and disclose. | BLOCKED: maintainer disposition |
| Documentation | Guide, README, public optional args, event/export contracts, source coverage and exact isolated validation commands; existing chart references only. | Planned |
| Graphify/Karpathy | Existing graph queried; truncated graph followed by direct source inspection. Extend the existing session and add one formatter module; no trainer refactor. | Completed planning review; graph refresh after future code changes |

Disposition is recorded in `verification.md`: implement observation only, retain current experiment behavior, and make unavailable evidence explicit. Leakage safety, input validation and evidence honesty remain mandatory. Any requested scientific remediation requires a separate revised scope.

## Project Structure

### Documentation (this feature)

```text
specs/002-readable-training-logs/
├── spec.md
├── plan.md
├── research.md
├── coverage.md
├── data-model.md
├── quickstart.md
├── contracts/terminal-and-evidence.md
├── checklists/requirements.md
├── tasks.md
└── verification.md                 # created during approved execution, not fabricated now
.specify/assessments/readable-training-logs/research.md  # post-plan evidence review
AGENTS.md                           # update only the marked plan link
```

### Source Code (repository root)

```text
pipeline.py                         # flag, event-driven progress/timing, safe terminal lifecycle
src/ai_job_market/
├── core.py                         # additive event/context calls at existing boundaries
├── training_audit.py               # existing session, independent sinks, scoped contexts/exports
└── training_console.py             # new small human formatter and synchronized block writer
src/training/                       # unchanged facades; compatibility checks
src/pipeline/                       # unchanged facades; compatibility checks
config/project.yaml                 # unchanged training settings
requirements.txt, uv.lock           # unchanged inherited edits

tests/
├── test_training_behavior.py        # extend parity; keep scientific characterization
├── test_training_audit.py           # file-only structured assertions and sink failures
├── test_training_audit_integration.py
├── test_training_fold_audit.py
├── test_training_audit_docs.py
├── test_training_console.py         # new depth/table/readability tests
├── test_training_stage_coverage.py  # new actual lifecycle/coverage tests
├── test_training_evaluation_logs.py # new comparison/tuning evidence tests
├── test_training_evidence_exports.py# new snapshot/manifest/CSV tests
└── test_training_cli.py             # new parser/progress/subprocess tests

docs/TRAINING_AUDIT.md
README.md
```

**Structure Decision**: Keep current import and serialization identities. No new package, generic event bus, async queue, schema framework, web endpoint or logging service. Do not edit segmentation helper algorithms; observe their existing caller boundaries and returned evidence tables.

## Phase 0 — Research Results

Decisions R1–R9 are resolved in [research.md](research.md). Findings most likely to invalidate a superficial plan:

1. Current tests require JSON on stdout; all such assertions must move to file evidence while preserving payload/call-count checks.
2. Missing train scores make an evidence-backed three-way fit diagnosis impossible within no-extra-prediction scope. Display `insufficient evidence` for today's candidates.
3. Current positional digest is not a dataset-content identity. Add a genuine DEV content/order fingerprint and preserve the legacy field's meaning.
4. `finalize_unobserved` must not fabricate PASS; real start/end events replace marker heuristics only in the presentation layer.
5. Source CLI has no workspace option. Validation must use clean copied projects, not root CLI execution.
6. Existing nested-tuning/selection/test-coverage governance gaps cannot be solved by prettifying text.

## Phase 1 — Design

### Session and terminal ownership

`TrainingAuditSession` remains the single source of audit records. `run_pipeline` gains optional keyword-only `debuglog=False` and `on_audit_event=None`, retaining all three positional arguments and the return dictionary. The observer is for CLI stage timings; it must not change model data or decisions. Core use without an observer still prints the complete readable progress.

A session assigns event sequence/context and writes the structured file independently of console rendering; one broken sink must not disable the other. The formatter returns human text with explicit depth 1–4 (indent 0/2/4/6 spaces). A synchronized block writer, reused by CLI heartbeat output, prevents line interleaving. No interactive redraw, root-logger mutation or terminal JSON fallback. Unclassified events use a safe human operation/message fallback; contract tests fail for unclassified production event families.

`on_audit_event` receives the normalized event after sink attempts; callback failures are observation warnings, not training failures. CLI does not duplicate stage printouts; it records timings and current-stage state only. The session always renders direct-core runs too. Existing preflight and final CLI summary remain readable but route through the same block writer. Preserve warning visibility and traceback behavior for actual training errors.

### Lifecycle and completion

Add a narrow session-owned observation context for stage/operation/evaluation/trial identity and guaranteed end/error recording around existing code. It may track open IDs, but may not wrap/replace estimators or alter return values. Started operations close on completion; exceptions record failed/cancelled and re-raise unchanged. Parent stages close only after their current writes and required operations return. No reordering to fit a pretty hierarchy.

Replace CLI save-function monkeypatching on the main execution path with stage events. Retain timing filenames and columns, with explicit `PASS`, `FAIL`, `CANCELLED`, `SKIPPED`, `UNKNOWN` semantics. A failed existing equivalence check must appear as failed evidence; do not change its calculation or the trainer's control flow. Run summary distinguishes scientific return/exit status from observation completeness. Unknown stages cannot be promoted to PASS by finalization.

### Data and model evidence

Use already-held inputs and returned outputs: safe fields from cleaning audit; actual raw/prepared/DEV/test counts; actual primary and experimental feature lists; fitted preprocessor names/vocabulary and transform shapes. Count formulas name their denominators and sequential exclusions. Fingerprint the ordered DEV frame for audit provenance only, not as model input or split policy.

Assign a unique evaluation ID per `evaluate_model_cv` invocation, with explicit parent stage and tuning trial context rather than display-name parsing. Reuse the already-created split arrays. Emit aggregate metrics after the existing summary is constructed, deriving log-only variability with `ddof=0` while leaving the returned tables unchanged. Importance-only fits retain `not_computed` metric status.

Comparison and tuning summarize all evaluated models/trials using current results; each fit assessment is currently `insufficient_evidence`. No additional scoring or automatic diagnosis. Show tuning progress on every completed trial at level 2, fold results at level 3, complete parameter/feature evidence at level 4. Do not replace final s1 selection with later tuning winners. Non-RF winners get a tuning-skipped event.

Branch A receives bounded substage events around its existing calls and table references; the coverage inventory explicitly excludes estimator/resample internals. Existing segmentation metrics are not salary fit diagnostics.

### Exports and file compatibility

Keep strict JSONL in the existing `.logs` filename, `schema_version=1`, existing fields and types. Add optional `detail_level`, `stage_id`, `parent_id`, `evaluation_id`, `trial_id`, `dataset_id`, `evidence_refs` and explicit missing-value reasons. Retain `level` as severity; it is not the new detail depth. Do not repurpose `features` from a list into a count; use `encoded_feature_count` on new events.

A unique sibling directory holds manifest and needed CSV snapshots. CSV fields and manifest rules are in the contract. Persist complete safe evidence whether debug is enabled or not; preview at most 20 data rows in each rendered table. Write the evidence before announcing a valid path. Avoid recursive calls through `core.save_csv` from the audit exporter. Reuse same-run immutable exports; snapshot relevant mutable model/tuning tables so later runs do not invalidate older logs. Do not copy raw data packs.

Read-back validation uses standard JSON/CSV readers and current-run manifest entries. No replay CLI or arbitrary external CSV import. New export errors warn, mark evidence incomplete and leave scientific control flow unchanged. Invalid/escaping destinations remain errors before an unsafe write, not silently accepted fallbacks.

### Failure semantics and minimal safety

Distinguish invalid destination validation from ordinary logging I/O failure: reject unsafe paths; continue with visible incomplete evidence for permission/disk/closed-sink failures where training itself remains valid. Independently guard serialization, rendering, callback, file creation/flush/close and final manifest/timing writes. Always restore session context and close owned resources without masking a training exception. Two failed sinks cannot guarantee delivery; document that limitation.

No new security feature is scheduled. Preserve existing input/path checks, exclusive run filenames and exclusion of secrets/raw row dumps. User request not to prioritize security does not require removing working safeguards.

### Agent context and post-plan review

`AGENTS.md` now links this plan inside its existing SPECKIT markers; its task-approval sentence and unrelated content were preserved. The installed script directory has no agent-context update script, so the required link update was performed directly rather than invoking a nonexistent command.

The requested post-plan research was completed with a read-only AST call-site cross-check and recorded in `.specify/assessments/readable-training-logs/research.md`. It reconciles direct fit/transform/predict/serialization sites with `coverage.md`, confirms the missing train-score evidence and retains the governance gates. No training/test execution is implied. The resulting 35-task breakdown is in [tasks.md](tasks.md).

## Phase 2 — Verification and Delivery Plan

1. Resolve governance and capture explicit task approval. Preserve inherited user edits; snapshot relevant baseline source hashes and an isolated working-tree copy before implementation. HEAD alone is not the correct baseline.
2. Characterize baseline behavior and full-run artifacts with fixed inputs; then write failing tests for each new presentation/export behavior. Use no-audit function sessions/spies for observation-free unit parity and a preserved pre-change working-tree copy for full-run parity.
3. Implement the session/sink foundation, then the default-readable US1 vertical slice. Add debug data evidence, model/tuning reporting and export/failure coverage in small verified increments.
4. Prove exact event lifecycle and coverage IDs against the inventory, not merely `event count > 0`. Include normal/debug, failed fold, non-RF tuning skip, undefined scores, repeated names, duplicate indices, shared periods, empty/small partitions, terminal/file failures and interruption.
5. Run focused and full pytest suites, current release-contract consumer checks, and isolated normal/debug CLI plus core workflows. No UI behavior changes are planned; import/artifact-contract regression tests cover affected consumers without introducing UI training.
6. Compare baseline/normal/debug semantic outputs: split membership, features, settings, model calls and metrics (`rtol=1e-9`, `atol=1e-9` initially); exclude timestamps, durations, source workspace strings and audit-only additions. Do not compare joblib bytes; compare loaded contracts and approved equivalent predictions.
7. Reconcile every displayed number with source events/CSV to declared display precision. Exercise the five-minute reviewer workflow. Record wall time, event/export sizes and limitations without claiming an unmeasured speed improvement.
8. Update guide/README and verification record, then refresh Graphify after code verification or document a valid inability. Mark tasks complete only with recorded evidence.

Exact commands and isolated CLI-copy workflow: [quickstart.md](quickstart.md). Tasks are generated after the requested post-plan assessment; they are not executed in this planning command.

## Complexity Tracking

| Violation / inherited constraint | Why observation-only handling is needed | Simpler alternative rejected because |
|---|---|---|
| VI.2 full candidate test reporting absent | User forbids extra evaluations; record unavailable cells and existing final-test evidence | Scoring all candidates changes held-out exposure and call counts; cannot be authorized by this feature |
| VI.3 nested tuning absent | Preserve the actual repeated DEV CV experiment and expose its limitations | Adding an outer loop changes training cost, calls and reported estimands |
| VI.4 R² tuning / missing variance-tie rule | Preserve rankings and final settings so logging is not a scientific migration | Silently switching metric/tie rules would violate parity |
| Historical test exposure / whole-data descriptive summaries | Report actual scopes honestly; do not claim unseen-test purity | A logging patch cannot retroactively make exposed data unseen; new leakage is not allowed |

These are proposed bounded treatments for maintainer review, not approved exceptions. No unresolved technical clarification is hidden behind these governance gates.
