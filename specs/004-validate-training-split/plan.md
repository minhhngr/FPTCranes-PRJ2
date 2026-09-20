# Implementation Plan: Training-Only Split and Validation Evidence

**Branch**: `006-validate-training-split` | **Date**: 2026-09-19 | **Spec**: [spec.md](spec.md)  
**Input**: `specs/004-validate-training-split/spec.md` and user's three clarification answers.  
**Status**: Implemented and fixture-verified after explicit approval. Real compliant execution remains blocked on verified future reserve data and exact partition approval.

## Summary

Add a training-only offline entrypoint that consumes existing cleaned raw artifacts and existing preprocessing definitions, validates strictly chronological whole-month TRAIN / EVALUATION_HOLDOUT / INFERENCE_RESERVE partitions targeting approximately 80/19/1, and publishes detailed English evidence. All five-family comparison, feature-family ablation and RF tuning happen inside TRAIN. Five expanding monthly outer folds replace the legacy sliding blocks **in this new entrypoint only**. The 19% is opened after configurations are frozen; the last 1% is inference-only and never scored here.

Preserve UI, source/preparation/preprocessing code and existing serving artifacts. New packs are not automatically consumed by current Pages 04–06. They expose an explicit read-only contract for future integration. No root-pipeline invocation, no dependency changes, no classifier training, no auto-deployment.

The old snapshot has no verified untouched future reserve. Implement fail-closed preflight and exercise training with isolated fixtures. Real artifact generation requires acceptable future inputs through the existing preparation process, not a manufactured split. An unavailable real run is a documented data limitation, not a reason to fake successful metrics.

## Technical Context

**Language/Version**: Python 3.13.12 in `.venv` (checked during planning); existing Ruff target remains unchanged.  
**Primary Dependencies**: pandas 2.3.3, NumPy 2.5.3, scikit-learn 1.9.1, Plotly 7.1.0, existing joblib and PyYAML; stdlib argparse/json/hashlib/pathlib/time. No additions/upgrades.  
**Storage**: Input `outputs/01_data_basic_clean/basic_clean.csv`; feature policy `config/project.yaml`; new immutable evidence `outputs/training_validation/<run_id>/`; new non-serving bundles `artifacts/training_validation/<run_id>/`; persistent exposure/access ledger `outputs/training_validation/holdout_access/`.  
**Testing**: pytest 9.1.1, small chronological fixtures, estimator/loader spies, reference metric calculations, CLI subprocess/contract tests and protected-file hashes.  
**Target Platform**: Linux offline CLI; no network/background service.  
**Project Type**: Additional offline training/reporting command sharing existing model/preprocessor factories.  
**Performance Goals**: At most 540 fits for a complete RF-winning experiment before reuse; prediction microbenchmark adds at most 1,784 calls on TRAIN features and zero fits. Zero fits/predicts/model loads in inspect/check/reuse paths. Publish measured fit/search cost, batch-specific p50/p90/p95, throughput, sizes and overhead; no invented latency SLA. Runtime protocol: [contracts/runtime-observability.md](contracts/runtime-observability.md).  
**Constraints**: No edits to `core.py`, `pipeline.py`, `config/project.yaml`, preprocessing/preparation modules/artifacts, UI, requirements/lock or existing model bundles. No reserve prediction or target exports. Fresh-data acquisition is external.  
**Scale/Scope**: Current snapshot 1,499 rows/15 observed months; new workflow requires at least nine observed TRAIN months and nonempty strictly later holdout and reserve. Five families, six ablation families, two final feature variants.

## Constitution Check

**Pre-research**: User clarifications resolve partition and classification semantics. Legacy non-nested/R² selection and historical exposure are incompatible with new compliant claims; do not inherit those behaviors.  
**Post-design**: Design meets new-path gates below. Historical input readiness is **blocked**, not waived. This is not a statement that legacy code or the current data is compliant.

| Gate | Design/evidence | Disposition |
| --- | --- | --- |
| Data integrity | Raw cleaned input; existing 13-feature allowlist; dates only for split; fresh fold-local preprocessing; strict whole months; reserve never scored | Design pass; leakage and target-loader spies planned |
| Reproducibility | Seed/versions/producer and input hashes, exact row membership, actual ratios, bounded trial matrix, final params | Design pass; no experiment executed |
| Artifact contracts | Separate namespace; explicit schema/method versions; safe immutable publication; old packs neither rewritten nor advertised as new evidence | Design pass |
| Verification | Failing-first tests per story, tiny real-estimator integration, actual entrypoint inspect/check, full regression suite in isolated workspaces | Planned |
| Streamlit boundary | No new UI imports/actions; no bundle promotion; existing pages remain on old experiment | Design pass; protected-file hashes and consumer checks planned |
| Input validation | Contained workspace/config/artifact paths, fixed raw input, month/feature schema, approved membership fingerprint, finite values and bounded resources | Design pass |
| Candidate ladder | Five families on same five CV folds and final holdout; never reserve | Design pass |
| Nested tuning/selection | MAE-based family freeze; RF search inside outer-training subsets; final search entirely inside TRAIN | Design pass with disclosed family-selection conditioning |
| Importance/uncertainty | Method-labelled encoded/raw/joint-family importance, subgroup support, descriptive holdout q90 | Design pass; no causal/calibration guarantee |
| Documentation | English 5W1H, metrics guide, traceable generated charts and stage conclusions | Planned |
| Known exposure | Current March records are exposed; full current-data run refused due to reserve requirement | Data gate BLOCKED; not a constitutional exception |
| Runtime and log integrity | Bounded TRAIN-only measurement, scientific vs operational status, raw samples and human/agent/UI parity | Design pass; new tests required |
| Graphify/Karpathy | Query and source follow-up; seven focused modules, unchanged legacy code, no service framework | Design pass |

## Project Structure

### Documentation (this feature)

```text
specs/004-validate-training-split/
├── spec.md
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   ├── training-evidence.md
│   └── runtime-observability.md
├── checklists/requirements.md
└── tasks.md
.specify/assessments/validate-training-split/research.md
AGENTS.md                              # marked plan link only
```

### Source Code (implemented, isolated offline workflow)

```text
src/ai_job_market/
├── training_validation.py              # CLI + ordered orchestration only
├── training_partitions.py              # metadata-only allocation, exposure/fold checks
├── training_evaluation.py              # paired metrics, ablation, importance, diagnostics
├── training_search.py                  # bounded nested stepwise RF search
├── training_runtime.py                 # timing contexts and bounded TRAIN-only microbenchmarks
├── training_evidence_io.py             # validation, provenance, events, lock, publication/read
└── training_report.py                  # pure English conclusions and offline Plotly figures
config/training_validation.json         # independent training-only policy; no dates auto-approved

tests/
├── test_training_partitions_v2.py
├── test_training_evidence_v2.py
├── test_training_evaluation_v2.py
├── test_training_search_v2.py
├── test_training_runtime_v2.py
├── test_training_report_v2.py
├── test_training_validation_cli.py
└── fixtures/training_validation/        # small synthetic chronological inputs, not real targets

docs/TRAINING_VALIDATION.md
README.md                              # new opt-in command and warning about old command
```

**Structure Decision**: Reuse `core.candidate_models` and `core.make_model_pipeline` without calling legacy CV/tuning/ablation/finalization orchestrators. Existing helpers import safely without executing a pipeline. Keep metrics with explicit undefined handling in the new evaluator because legacy R² defaults may force finite outputs. Existing `training_audit` has a primary-pipeline-specific path/stage contract; leave it unchanged and use a small new JSONL writer, not a reusable event platform. No classes/frameworks beyond simple typed records where necessary.

## Phase 0 — Research and resolved decisions

[research.md](research.md) fixes the proposed month-allocation objective, five outer/three inner folds, selection/tie rules, fit-diagnostic heuristic, six feature families, stepwise search bounds, two-feature assumption and diagnostic methods. These are design proposals subject to task review, not outcomes inferred from holdout scores.

Unseen reserve provenance cannot be generated by software from historical data. A dataset custodian provides source lineage/exposure attestation; code reconciles known evidence and rejects contradictions. Unknown reserve provenance blocks execution. Full correctness testing uses controlled fixtures; real validation may end at preflight until fresh data is available.

## Phase 1 — Design and lifecycle

### 1. Inspect and approve the data declaration

`python -m ai_job_market.training_validation inspect --workspace ROOT` is read-only, fit-free and prediction-free. It validates fixed input paths, metadata, feature policy and configured bounds; computes a target-independent month split proposal and expected fit budget; reports actual ratios and lineage limitations as JSON plus English console text. No dependency installs or preparation subprocesses occur.

Approval JSON freezes source hash, exact split/boundaries/counts, policy hash, reviewer/date, exposure evidence references and optional business MAE/RMSE limits. Date cutoffs may be user-selected but must be whole-month, contiguous, cover all eligible rows and preserve strict ordering. All ratio deviations are visible for approval; no hidden same-month allocation or silent row dropping. A changed input invalidates approval. Default acceptance limits are absent, so no automatic Good result.

Source identity uses immutable input hash plus source row ordinal; cross-file known-exposure checks also use stable normalized feature/month fingerprints excluding targets, with ambiguous duplicate matches treated conservatively. Input provenance validation compares the known prepared snapshot and available historical manifests. It never proves absence of unrecorded external use. Exposure status is attested-unexposed / known-exposed / unknown, with citations and explicit assurance limits. Unknown or exposed reserve is a hard block.

### 2. Train and compare using TRAIN only

`run --workspace ROOT --approval FILE` validates everything before fitting. It writes a staging run and emits stage lifecycle events. Schema validation applies to TRAIN targets first; HOLDOUT targets are unavailable to selection/tuning code until the frozen lock. Reserve targets are not materialized into evidence or passed to training interfaces. Prefer column-projected metadata readers and explicit partition-scoped target loaders; whole-file hashing is allowed but target statistics/inspection outside authorized scopes are not.

Model comparison produces 25 paired train/validation fold records and separate equal-fold and pooled summaries. Timing clearly separates fit-including-preprocessing, train prediction and validation prediction; no false claim to isolate preprocessing time if the pipeline does not measure it. Ranking uses unrounded MAE and the documented overlap/simplicity rule. The five-factory comparison remains distinct from tuned-model evaluation. Failed or incomplete candidates block a selection declaration.

Ablation uses only approved feature subsets and fixed RF settings. RF fold importance is captured during existing comparison fits, not regenerated solely for visualization. Explain pairwise drift and lack of causality. No diagnostic silently changes the full feature set or the fixed two-feature pair.

### 3. Freeze family, tune conditionally and compare variants

Write family selection before tuning begins. RF receives five outer-training searches and a final all-TRAIN inner search, each with 26 bounded slots on three temporal folds. Stage winners carry forward; cache exact identical configurations/folds/feature policies only. Fit counts/progress log completed, failed and reused slots separately. Non-RF family uses frozen defaults and an explicit tuning-skipped record.

Full and Top-2 outer comparisons use the same outer-training-selected parameters in each fold. Final all-TRAIN fitted bundles use the final search parameters and never include the holdout or reserve. The Top-2 result is a portability/feature-reliance comparison, not a new holdout-selected champion. Final family stays frozen regardless of the later holdout ranking.

### 4. One evaluation transaction and failure handling

Before opening holdout targets, serialize an immutable declaration containing candidate configurations, final full/Top-2 bundle hashes/features, memberships, selected family, diagnostics, thresholds and expected outputs. Acquire an exclusive local filesystem claim under `holdout_access/<holdout_id>/`, where `holdout_id` identifies source row membership independently of arbitrary run ID. Record `started` before target access.

- Completed matching evaluation: reuse verified outputs, no fit/predict.
- Completed evaluation with changed declaration: reject fresh final evaluation for that population; require new future holdout data. Do not erase exposure by changing run IDs or data container hashes; reconcile known row fingerprints.
- Interrupted/failed after lock: mark exposure as started/failed; do not silently retry target evaluation. Preserve partial evidence for audit, do not publish a complete pack. A new independent evaluation requires new data; recovery of already written verified results is allowed without recomputation.
- Historical exposure from prior systems: must be declared as historical before the run. A first transaction in this workflow may publish historical diagnostics only when reserve requirements are satisfied; it cannot become a pristine-test claim or Good recommendation.

This is local workflow enforcement, not security against an operator deleting the ledger or copying data to another machine. Record this limitation. No hidden force/unlock switch.

After lock, fit any remaining frozen factory models on TRAIN, then score the five families plus full/Top-2 on the holdout. No new parameter choice is allowed. Final diagnostics/importance/subgroups/q90 use holdout only and are labelled accordingly. No predictions, errors, permutation operations, intervals or coverage are computed on the reserve.

### 5. Publish results, evidence and next action

Strictly validate all required tables, statuses, hashes and bundle metadata before atomic publication. No mutable pointer to the new model is added to the existing UI. Consumers explicitly load a run ID through the new read-only validator. Existing source/producer/schema identities must match for reuse; changed artifacts are new experiments or marked unavailable, never renamed historical evidence.

English report order: 5W1H/data contract → split/fold walkthrough → five-family R²/MAE analysis → runtime → selected family → ablation/importance → stepwise tuning → full/Top-2 CV → frozen holdout evidence → residual/subgroup/uncertainty limitations → overall decision. Every stage states Finding, Evidence, Limitation and Next action.

Statuses are separate:
- Execution: planned, blocked, running, failed, cancelled, complete, reused.
- Scientific outcome: Good, Bad, Inconclusive, Blocked. A preflight/contract failure is Blocked. Complete valid metrics that fail configured absolute error limits or fail to beat Dummy holdout MAE produce Bad. Missing business limits, undefined required metrics, or historical holdout exposure prevent Good and produce Inconclusive unless already Bad. Good requires complete valid evidence, applicable MAE/RMSE limits supplied before training, both limits met, better holdout MAE than Dummy, and eligible exposure provenance. Good is the scientific prerequisite for human review, not production fitness. Combined deployment-review eligibility additionally requires configured and passing operational limits from the runtime contract. Missing operational limits remain not assessed. Fold fit labels are independent of these run outcomes.

Monitoring/retraining guidance identifies data owner and future labelled monthly batches, requires no reserve scoring here, and recommends a fresh later holdout after failed final evaluation. No service, scheduler or deploy action is implemented.

### 6. Runtime, extended metrics and dual-audience evidence amendment

The previous plan specified basic fit/predict time but did not define latency workloads, samples, cost accounting, resource scope or a concrete human/agent/UI parity contract. The amendment fills those gaps without rerunning models for review. [contracts/runtime-observability.md](contracts/runtime-observability.md) is authoritative for these additions.

Add `training_runtime.py` as a small measurement helper, not a profiler platform. Reuse existing fits: benchmark the 25 candidate-fold pipelines while they are resident and the two final full/Top-2 bundles before holdout access. Use raw TRAIN features only, batch sizes 1 and min(100, TRAIN rows), three warmups and 30 measured calls. Keep every sample and environment/thread context. Record final bundle first-load/first-predict observations honestly as in-process, uncontrolled-cache measurements, not cold production startup. Maximum 1,784 extra predict calls; no extra fits, reserve work or benchmarking on an unchanged reused pack. Tuning trial costs are instrumented but trial models are not all microbenchmarked.

Capture wall/CPU spans, search/fit/predict cost, benchmark overhead and artifact size. CPU/RSS are explicitly process-scoped; never attribute the lifetime peak to one estimator or sum nested timings as total runtime. Numerical-library threads are bounded with the existing scikit-learn threadpool dependency; unknown effective settings make speed comparisons non-comparable. No timing-based assertions that become flaky under CI; test clocks/arithmetic deterministically and use tiny estimator smoke tests for integration.

Derive bias, tail errors, baseline-relative skill, fold gaps and matched tuning/feature-variant deltas from existing predictions. Save OOF row evidence so pooled metrics are independently checkable. No new scoring populations or family-selection metric. Report MAE quality winner, tie-aware selected family, fastest comparable candidates and a descriptive accuracy/fit-time Pareto chart as different results. Optional runtime/size budgets feed operational status, not scientific family selection.

Publish `training.log`, `agent_summary.json`, `metric_catalog.json`, `ui_summary.json`, OOF rows, runtime sample/summary tables, performance/tradeoff tables and an operational assessment under the same new pack. One event source feeds the English transcript and structured summaries; prose is not an API. Correlation IDs include invocation and operation so resumed inspections/reuse are not mistaken for new original-run events. Immutable packs stay immutable; reuse emits a new invocation summary without rewriting original evidence. Missing/corrupt telemetry prevents a complete publication.

The UI snapshot is a complete-pack read-only index containing typed KPI/table descriptors, source references, units/precision and availability reasons. It does not embed unbounded logs or data rows, load models, read staging, start measurement or modify current pages. Future UI integration still needs separate approval. Agent/human/UI summary parity is verified using fixture consumers against numeric sources; no source-code or English parsing is needed to understand conclusions.

### 7. Explicit fold walkthrough and per-model conclusions

The fold-count request means “Show how the folds are split and how many rows each contains.” Basic memberships were already required, but are not sufficient for a readable UI-facing explanation. Contract sections 9–10 add mandatory `fold_summary.csv`, `monthly_row_counts.csv`, `fold_explanation.md` and `model_conclusions.json` to the existing new pack. These derive from existing declarations/results; no new fits, predictions or dependencies are needed.

`training_partitions.py` derives one summary per unique fold with exact month lists/counts, parent population, training/validation/later-unused row counts, explicit percentage denominators, newly added history and actual chronology/overlap/expansion checks. Independently reconcile monthly totals against row memberships. `training_evidence_io.py` exports summaries and emits `fold_plan_defined`/`fold_plan_used` references; reused folds do not inflate unique coverage. Inner summaries name their outer-training parent; final-inner folds name the full TRAIN population. Skipped RF tuning is not applicable, never fabricated executed folds.

`training_report.py` supplies an English explanation and timeline/table descriptors for future Page 04, before comparison results, and grouped nested-fold details for future Page 05, before tuning results. Explain why all earlier months remain in training, why prior validation rows can enter later training, and why the later holdout/reserve are excluded. Show exact actual counts rather than fixed 200-row examples or hard-coded percentages. Encoded column dimensions are not row counts.

Produce mandatory evidence-backed conclusions for all five candidate roles and separate final-full/Top-2 roles. Every record carries accuracy/fit/stability/runtime evidence, valid folds, selection/comparison reason, limitations and safe next action. Losing, failed or unavailable candidates stay visible with reasons. Dummy is the baseline; RF is not assumed best; holdout findings cannot rewrite CV selection. A complete experiment has seven role records even when fits are reused. These records are referenced consistently by the English transcript, agent summary and UI snapshot.

The immediate deliverable remains offline evidence, not changed UI pages. Existing tasks for fold declarations, report/consumer parity, documentation and comprehension are expanded rather than adding a new implementation phase.

## Phase 2 — Verification and delivery

1. Record task approval, preserve dirty files and capture protected-file/artifact hashes. Do not commit or modify inherited untracked `uv.lock`.
2. Implement US1 preflight first with failing chronology, exposure and fold tests. The actual snapshot should fail unseen-reserve readiness without fitting; that is an expected result.
3. Implement US2 comparison on fixtures with metric arithmetic, estimator-fit membership and deterministic ranking checks. Capture progress/conclusions from the beginning, not as a later scientific retrofit.
4. Implement US3 search/lock/final diagnostics using fit/target-access spies and tiny bounded estimator fixtures. Negative cases cover hidden future data, stale approvals, undefined metrics, partial publication and changed declarations after exposure.
5. Complete US4 English report/glossary/chart contract and read-only consumer tests. A UI-facing pack is tested without touching UI code or activating models.
6. Exercise the real new CLI in temporary fixture workspaces, then read-only current-data preflight. Run all relevant existing tests; any producer-heavy tests use isolated workspaces. Do not execute root `pipeline.py`.
7. Regenerate only affected new-namespace evidence if input readiness passes and training tasks are approved; otherwise record explicit new-experiment unavailability in verification notes. Old outputs remain valid only for their old experiment. Prove unchanged-run reuse with spies.
8. Reconfirm no protected input/preparation/UI/bundle hashes changed, review the diff and record test results/limitations in `verification.md`. Update Graphify after source changes or document a valid exception. Documentation-only planning does not require graph regeneration.
9. Verify benchmark raw-sample math/call bounds, timing/RSS semantics, target-free workload membership, runtime comparability and zero rebenchmark on reuse. Check typed English/agent/UI snapshot parity, stable error/progress codes and missing-budget behavior. Include fastest-versus-selected interpretation in the human walkthrough and structured-agent fixture test.
10. Verify exact fold-count reconciliation using unequal monthly sizes, month gaps and nested parents; count mutation must invalidate summaries/snapshots. Require five candidate plus two final-role conclusion records, including missing/failed states. Human and agent acceptance must explain fold 3's months/row counts, expanding history and each model's decision from exports alone.

## Complexity Tracking

No new-path constitutional exceptions are requested. The additive CLI/namespace avoids breaking primary-pipeline and UI consumers while changing scientific semantics. A shared-core replacement is rejected because it would silently affect preprocessing/segmentation/UI experiments outside the approved scope. Historical data readiness is not waived; unavailable real-run evidence stays unavailable.

## Review decisions and risks

- Task approval includes the proposed deterministic split-ranking objective, overlap/simplicity rule, diagnostic threshold, fixed Top-2 pair, 540-fit upper bound and added bounded runtime/logging contract (up to 1,784 TRAIN-only prediction calls). These are pre-run policy choices, not claims of optimality.
- Real unseen reserve data is not available in inspected artifacts; data acquisition/preparation is a separate responsibility. Approving implementation does not authorize fabricating data or overriding this gate.
- Ratio closeness is data-dependent; a near-1% whole-month reserve may require a larger or differently sampled future dataset. Actual allocation requires explicit approval before running.
- Existing pages continue displaying legacy evidence until a separate UI-integration task is approved. New exports are available to future consumers but are not automatically shown.
