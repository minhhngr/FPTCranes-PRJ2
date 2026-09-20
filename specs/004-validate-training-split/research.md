# Research and Design Decisions: Training-Only Validation

**Date**: 2026-09-19  
**Status**: Design for task review; no implementation or training performed.  
**Evidence research**: `.specify/assessments/validate-training-split/research.md`

## 1. Resolved user decisions

The user's clarification makes the 19% a final evaluation holdout, with all selection/tuning/CV inside the 80%. The 1% is strictly later inference-only data. Whole-month chronology overrides exact proportions. Classification coverage is educational; the target remains continuous salary regression. These replace the three original open questions.

## 2. Existing behavior and reuse boundaries

| Finding | Evidence | Decision |
| --- | --- | --- |
| Legacy CV slides one 200-row block rather than expanding by month | `src/ai_job_market/core.py:1040–1082` | New training-only splitter; do not modify the shared legacy function or reinterpret its old metrics |
| Existing split is 1,201 DEV / 298 March holdout | `outputs/02_data_ready_for_ml/temporal_split_summary.json` | New target-independent membership declaration; preserve upstream files |
| Latest month has 298/1,499 rows, already evaluated | monthly split counts; `specs/003-improve-model-ui/plan.md` | Existing snapshot cannot meet an unseen future-reserve claim; complete run blocks rather than relax chronology |
| Existing preprocessing builder supports feature subsets | `core.py:997–1037` | Reuse builder with fresh fold-local fitted state, not encoded CSVs |
| Five candidate estimators already exist | `core.py:1405–1416` | Reuse `candidate_models(seed)` and freeze actual parameters in manifest |
| Legacy family ablation includes prohibited `experience_level` | `core.py:1445–1490`; `config/project.yaml` | New leave-one-approved-family-out diagnostics only; do not call legacy ablation orchestrator |
| Old RF search ranks R², fixes depth 20 in later stages and applies initial-grid winner | `core.py:3194–3583` | Keep five-step narrative/search values, but use MAE, carried winners and nested temporal evaluation |
| Existing audit session writes primary-pipeline logs and requires its historical stage set | `training_audit.py:113–243` | Do not extend/break that public lifecycle. New small event writer in the new evidence namespace, not a generic logging framework |
| Existing paired evaluator demonstrates useful metrics/timing patterns | `ui_evidence_training.py:24–60` | Follow the pattern; new evaluator owns explicit folds, undefined-metric handling and new event identities |
| Existing two-input feature contract is category + experience years | `ui_evidence.py:47`, `salary_inference.py:54` | Propose same fixed pair; no UI/serving model replacement |

Graphify was queried in both planning sessions. Latest query returned 164 nodes with 31 displayed; source inspection followed because the graph output was truncated. No hidden graph edges are assumed.

## 3. Outer partition proposal and approval

Work over fixed `outputs/01_data_basic_clean/basic_clean.csv` in a workspace, with the unchanged `config/project.yaml` feature policy. Read month/identity columns for planning; the reserve's target is not used, exported or passed to any model. Hashing source bytes for provenance is not target-aware analysis.

Enumerate pairs of boundaries between observed months, making nonempty contiguous TRAIN, EVALUATION_HOLDOUT and INFERENCE_RESERVE populations. Require at least nine observed training months so the earliest of five outer validation months has at least four earlier months for three inner folds. Rank proposals by sum of squared deviations from `(0.80, 0.19, 0.01)`; break ties by absolute training-share deviation, then absolute reserve-share deviation, then earlier boundary dates. All arithmetic uses exact row counts until final display. Use no target values or model metrics to choose boundaries.

Preflight prints the best proposal, all actual counts/shares/deviations, month counts, exposure status and a fingerprint. It never silently accepts a large ratio deviation: execution requires an operator-approved declaration of that exact fingerprint/boundaries/shares. There is no invented universal tolerance. A human can choose another valid whole-month pair in the declaration; validation checks its identity and all invariants. New source bytes invalidate old approval.

With the inspected counts, the natural last-two-month split is 895 training / 306 evaluation / 298 reserve (59.71% / 20.41% / 19.88%), not approximately 1% reserve in ordinary language. This is an illustrative count calculation, not an approved allocation. More importantly, those reserve records are already exposed. The implementation must report the blocker on current data. Future data must be supplied through the existing preparation process outside this feature; no row slicing, synthetic reserve fabrication, date changes or reassignment can fix historical exposure.

Exposure declaration is reviewer-attested evidence, not cryptographic proof of absence of all prior use. Reconcile known historical membership/fingerprints and local access ledger; contradiction wins over an assertion of “unseen”. Unknown reserve provenance blocks execution. Historically exposed holdout data may support clearly labelled historical evaluation but cannot yield a Good recommendation. Unknown holdout provenance blocks final target access pending review. Tests use known isolated synthetic fixtures, not claims about real unseen records.

## 4. Expanding monthly CV

Sort by year/month and stable source position. For a training population with M observed months, outer validation uses the last five observed months; each fold trains on **all** earlier observed months. Months without records are logged as gaps, not manufactured. Validation months are disjoint; a prior validation month's records legitimately join a later fold's training history. Strictly require max train month < validation month.

For RF inner selection, use the last three observed months within each outer-training subset and all earlier months for each inner fit. A final RF selection search uses three such inner folds within the full TRAIN population, never the 19% holdout. Five folds alone need six observed months; this experiment's nested readiness gate requires nine training months before any candidate fitting, so the budget does not depend on the eventual winning family.

## 5. Family selection and fit diagnosis

Primary ranking: arithmetic mean of five validation MAEs (equal fold weight). Report population SD (`ddof=0`), row counts and separate pooled out-of-fold metrics; pooled scores do not select the winner. Let b be the minimum-mean candidate. The overlap set comprises candidates whose `[mean-SD, mean+SD]` MAE interval overlaps b's. Prefer the simplest in this set using declared order: Dummy Median, Linear Regression, Ridge Regression, Gradient Boosting, Random Forest. This is a deterministic engineering heuristic, not a confidence interval or significance test. Show raw rank and tie-aware selected family separately. Runtime is evidence, not a hidden tie-break.

Fit labels are educational heuristics, not diagnoses guaranteed by R². For each fold, define `train_gain = 1 - model_train_MAE / dummy_train_MAE`, `validation_gain = 1 - model_validation_MAE / dummy_validation_MAE`, and `gap = (model_validation_MAE - model_train_MAE) / dummy_validation_MAE`. If denominators are zero, label inconclusive. Overfitting indication: train_gain > 0 and gap > 0.20. Underfitting indication: train_gain <= 0 and validation_gain <= 0. Good-fit indication: validation_gain > 0, valid validation R² >= 0 and gap <= 0.20. Otherwise inconclusive. Dummy is labelled baseline, not judged against itself. Aggregate label requires the same indication in at least four of five folds, else mixed/inconclusive. The 0.20 threshold is a declared proposal for task review, not a measured optimum. Publish values and rules so readers can disagree without losing the evidence.

## 6. Bounded stepwise Random Forest tuning

Run only if RF is the tie-aware selected family. Preserve these stage bounds from the existing workflow:

1. Four anchors `(trees, leaf, max_features, depth)`: `(300,2,0.7,20)`, `(100,2,0.8,null)`, `(80,1,0.8,null)`, `(200,1,0.7,20)`.
2. Trees: `[50,100,150,200,250,300]`.
3. Depth: `[10,15,20,25,30,null]`.
4. Leaf: `[1,2,4,8]`.
5. Max features: `[0.5,0.6,0.7,0.8,0.9,1.0]`.

26 trial slots/search, three inner folds/trial. Carry actual prior-stage winner's other parameters. Each stage considers the incumbent as well as that stage's values so it cannot silently discard a better already measured setting; an incumbent matching a previous result is reused on identical folds and does not consume extra fits. In stage 1 choose by lowest inner mean MAE; ties use fewer trees, shallower depth (null last), larger leaf, smaller max_features, then stable trial order. Apply the same rule in later stages. Final parameters come from the last carried winner, not the first legacy tuning table. No new dependencies or unbounded search.

Five outer-training searches provide outer held-out evaluation of this fixed RF tuning procedure. Their winners fit only their own outer-training rows. Apply each outer winner unchanged to the full and fixed two-feature variants for paired outer validation; do not search the Top-2 variant separately. The final three-fold search on all TRAIN determines the two final bundles' common estimator settings. Refitting uses all TRAIN only. For non-RF winners, use the frozen default estimator for full/Top-2 fold and final comparisons, with tuning explicitly skipped.

Family selection reused outer fold evidence, so nested RF results are conditional on a family chosen on this dataset: do not claim an unbiased nested estimate of the entire family-selection pipeline. The final holdout remains the independent assessment only when its exposure provenance supports that claim.

Worst-case upper fit bound: candidate comparison 25 + six RF family removals × five folds 30 + six searches × 26 slots × three inner folds 468 + paired outer-variant fits 10 + five candidate final fits and two final variants 7 = **540 fits** before exact-identity reuse. Fold RF importance is captured from existing fits. Permutation operations perform predictions only. Persist expected/actual counts and per-stage timing; do not assert a wall-clock SLA from this bound. Fixed seed 42 by default and `n_jobs=1` make hardware/runtime comparisons interpretable.

## 7. Ablation, importance and diagnostics

Six approved feature families: Job domain (`job_title`, `job_category`); Experience/education (`years_of_experience`, `education_required`); Geography (`city`, `country`); Company/work (`remote_work`, `company_size`, `industry`); Demand/benefits (`demand_score`, `benefits_score_10`); Skills (`required_skills`, `skill_count`). Leave one family out from the full approved set using frozen factory RF and the same five outer folds. Baseline is its already measured full-feature result. Ablation is explanatory, not automatic feature selection.

Capture RF encoded impurity importance during candidate fits; aggregate to raw columns and the six families using the fitted transformer's mapping. Store vocabulary-presence flags; raw-family adjacent-fold drift is half the L1 difference between normalized family-importance vectors (0–1). Encoded absences remain absent, not fabricated observed zeros.

After the evaluation lock, report each candidate and full/Top-2 variant's holdout metrics. For final variants, save actual/predicted/residual (`actual - predicted`), absolute error and raw-column permutation MAE increase with 12 repeats, seed 42. Also permute all columns in each approved family jointly using the same row permutation per repeat; this is distinct from permuting each column independently. Encoded impurity is available for tree models, absolute coefficients for linear models under a different method label; Dummy has no meaningful estimator importance. Do not substitute zeros.

Subgroup slices: job category, country and experience years `[0,3)`, `[3,7)`, `[7,+inf)`; missing/invalid values fail input validation, not silent recoding. Report counts/MAE/RMSE/R²/MedAE with support <20 flagged as small sample. No subgroup controls selection or promotion. Holdout q90 uses linear-interpolated 0.9 quantile of absolute errors; band is `prediction ± q90` without hidden clipping. Coverage on the same holdout is descriptive, not calibrated, and reserve coverage is never computed.

## 8. Runtime/logging specification review

**User signal**: Follow-up asks for more detail on runtime/best performance, logs readable by humans and agents, and information usable by the UI. This is approval to revise specifications, not to implement or benchmark.

**Gaps found in the draft**: FR-008 and the original runtime ledger only described basic fit/prediction timing. They did not define repeated-latency workloads, percentile samples, throughput units, cold/warm distinctions, inclusive timing accounting, resource scope, typed UI KPI payloads or consistency across narrative and machine summaries. Existing logs had broad event semantics but lacked bounded payloads, invocation correlation, progress denominator rules and operational-budget states. [source: prior versions of this feature's spec/plan/contract, reviewed in this session]

**Decision**: Add [contracts/runtime-observability.md](contracts/runtime-observability.md). Use existing mandatory fits for bounded prediction-only microbenchmarks on TRAIN; do not repeat fits to get a nicer runtime chart. Measure 30 calls after three warmups for batch 1 and capped batch 100, retain raw samples, report local p50/p90/p95 and throughput, and disclose first-load OS-cache limitations. No extra dependency installs, holdout benchmark or reserve inference. Up to 1,784 additional predict calls are a reviewable overhead budget, not measured runtime.

Derive extra regression bias/tail/baseline/gap metrics from existing predictions and persist OOF rows. Distinguish best accuracy, selected family, fastest measurements and a descriptive accuracy/fit-time frontier. No runtime-weighted winner, automatic optimization, classifier or holdout-driven tuning. Optional operational limits remain separately assessed; no latency thresholds are invented from this snapshot.

Use one event source for English transcript/narrative and structured agent/UI summaries with exact numeric provenance. Agent consumers rely on metric IDs/units/status enums, not English parsing; UI consumers receive complete-pack descriptors, not a live training log stream. No UI source changes are proposed.

**Evidence status**: These are protocol/design choices. No runtime timings, memory observations, fastest model or optimization gains have been measured in this review. Actual measurements require approved implementation and eligible inputs; tiny isolated fixtures validate instrumentation first.

## 9. Fold explanation and per-model conclusion review

**User signal**: “cần thể hiện cái cách chia fold như thế nào bao nhiêu dòng” means “Show how the folds are split and how many rows each contains.” The user also requests UI-ready detail and conclusions for each model.

**Finding**: FR-005, the membership contract and basic log template already required dates/counts, but did not explicitly require a readable fold-summary view, monthly count reconciliation, added history, parent denominators or a conclusion record for each non-winning model. Broad “section conclusions” could leave losing models unexplained. [source: this feature's reviewed spec and runtime contract]

**Decision**: Contract sections 9–10 add a reconciled fold summary/month-count table, English fold walkthrough and per-model conclusion artifact. Display-ready descriptors cover five outer folds for Page 04 and parent-labelled inner folds for Page 05. All five candidates plus final full/Top-2 receive distinct scoped conclusion/status records, including explicit failure/unavailable reasons. Counts and verdicts derive from saved memberships/results; no fixed example numbers become purported real evidence.

Existing partition/report/parity tasks are extended; total remains 51. This is a documentation-only refinement with no new fits, benchmarks, preprocessing or UI code. Actual page integration remains a separate approval. No claim is made that existing pages already render this new contract.

**Implementation-start Graphify review (2026-09-19)**: `graphify query "training validation fold summary training audit model conclusions UI evidence consumers" --budget 1200` found 278 nodes and displayed 34 before truncation. It confirmed the additive boundary around `core.py`, primary `training_audit.py`, `ui_evidence_io.py`, existing Page 04–06 consumers and conclusion builders. Direct source inspection remains required for truncated relationships. Simplest viable approach remains focused new modules and a new evidence namespace; do not alter legacy CV/tuning/UI contracts.

## 10. Rejected alternatives

- Random 80/19/1 split or splitting March into 19%/1%: violates strict monthly chronology and cannot undo exposure.
- Automatically rerun root `pipeline.py`: touches data preparation/segmentation and unrelated producers.
- Change shared `core.temporal_cv_splits`: would invalidate old UI/segmentation consumers beyond scope.
- Reuse old CV/tuning numbers under new labels: different experiment; scientifically invalid.
- Add classification estimators or salary thresholds: user explicitly excluded this.
- Force Random Forest, select Top-2 from test importance, or tune on the 19%: violates the agreed selection boundary.
- Activate new model in Page 06: UI/serving integration is separately scoped.
- Silently preserve non-nested R² tuning: conflicts with current constitution. Correctness changes stay in the new training path, with old artifacts retained as historical.
