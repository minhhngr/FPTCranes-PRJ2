# Data Model: Training-Only Validation Evidence

**Schema**: `training-validation/v1` (new namespace, not a replacement for historical UI evidence).  
**Target**: continuous `annual_salary_usd`, USD/year. No classification labels.

## Identities and relationships

| Entity | Required attributes | Invariants |
| --- | --- | --- |
| Dataset | relative input path, SHA-256, row count, ordered columns/schema, observed months, feature-policy hash | Exact input identity; no recleaning, no encoded training input |
| Row membership | dataset_id, row_id, source_position, period, partition, non-target fingerprint | row_id binds input hash + source ordinal; each row in exactly one outer partition; identifier never a feature |
| Exposure record | population identity, status, evidence refs/hashes, custodian, declaration time, known-history matches | Attestation is labelled as such; known exposure overrides assertion; reserve must be attested-unexposed with no known contradiction |
| Partition approval | dataset_id, policy hash, train_end, holdout_end, actual counts/shares, proposal hash, reviewer/date, exposure refs, acceptance limits | All whole months; train < holdout < reserve; nonempty; target-independent; altered inputs invalidate approval |
| Fold | fold_id, scope (outer/inner/final-inner), parent_fold_id, train/validation row IDs, period bounds, counts | Inner membership entirely inside applicable outer TRAIN; five outer/three inner; no shared months within fold |
| Feature variant | variant_id, ordered raw features, family map, blocked-feature policy | `full`, fixed `top2`, six removal variants; subset of approved columns only |
| Model configuration | configuration_id, family, full estimator params, seed, variant, preprocessing policy | Never inferred from a label; clone/re-fit state per fit scope |
| Trial | search_id, stage, trial_id, configuration_id, fold IDs, carried incumbent, timings, reuse reference, status | At most 26 slots/search; terminal status and all fold metrics linked; final winner actually applied |
| Fit evidence | fit_id, model/config/variant/fold, membership hash, dimensions, fit/predict durations, seed | Measured times are finite nonnegative; reused fit distinguished from a fresh measurement |
| Metric | fit/model/partition/fold, metric, value, unit, reason | MAE/RMSE/MedAE >=0; R² unbounded below and <=1 within numerical tolerance; undefined value null with reason |
| Selection | ranking scores, overlap set, simplicity order, chosen family, input evidence IDs, frozen_at | Only outer TRAIN CV evidence; independent from later holdout ranking |
| Evaluation declaration | declaration_id/hash, selection, all model/bundle/features/membership hashes, metrics/diagnostic policy, thresholds | Frozen before holdout access; changing it cannot reset exposure |
| Access ledger | holdout_id, declaration hash, run_id, timestamps, state, completed evidence pointer | Atomic exclusive claim; monotonic started -> complete/failed; never deleted to gain a new unseen test |
| Bundle metadata | model_id, bundle hash, fitted population ID, feature order, params, preprocessing schema, method version | Fit scope TRAIN only; not a serving activation; full/top2 distinguished |
| Conclusion | section, finding, evidence refs, limit, next action, status/rule version | Numeric statements derived from matching evidence; no unsourced optimum/causality claims |
| Run manifest | schema/method version, run_id, provenance, declaration, files/hashes, section availability, execution status, scientific outcome | Complete only after all required evidence validates; no reserve targets/predictions |

## Core table relationships

- `partition_membership.csv`: one row per eligible source record. Contains only identifiers, periods and partition labels, including target-free reserve references.
- `fold_membership.csv`: many rows per source record across folds, with fold ID, parent ID, role and row ID. A previous validation record may appear in later TRAIN history; this is intentional expansion, not leakage.
- `fold_summary.csv`: one row per unique outer/inner/final-inner declaration; exact month lists, parent counts, train/validation/later-unused rows, added history, explicit percentage denominators and computed overlap/chronology/expansion checks. Trials/models reference the same declarations rather than duplicating unique coverage.
- `monthly_row_counts.csv`: dataset/split/period/outer partition/row count/membership reference; summing the relevant month lists must reconcile with fold membership counts.
- `fold_explanation.md`: derived English split overview, five-row outer table/timeline and grouped nested-fold details. Unknown or skipped execution is explicit, not invented.
- `model_conclusions.json`: seven role records for a complete run (five candidates and final full/Top-2), with model/configuration/fold/population identities, status, findings, metric references, selection/comparison reason, limitation and safe next action. Reused fits do not collapse distinct reporting roles.
- `candidate_fold_metrics.csv`: 25 rows, one per candidate × outer fold; wide paired train/validation metric columns and timing references.
- `variant_fold_metrics.csv`: ten rows for full/top2 × five outer folds. For RF, configurations may differ across outer folds because tuning occurs independently inside each outer-training subset.
- `tuning_trials.csv`: slots and inner-fold metrics linked by search/trial/fold IDs; no rows for RF search if a different family won, accompanied by a skipped-section reason.
- `holdout_metrics.csv`: five frozen factory candidates plus final full and top2, with explicit role/configuration ID even when some share/reuse a fit.
- `holdout_predictions.csv`: final full/top2 predictions only; row ID, model ID, actual, predicted, residual (`actual - predicted`), absolute error and empirical band. No reserve rows.
- `permutation_importance.csv`: model/feature-or-family/repeat/scoring/MAE increase; negative values valid.
- `encoded_importance.csv`: model/fold/encoded feature/raw mapping/presence/method/value. Missing vocabulary is not observed zero.
- `importance_drift.csv`: RF adjacent outer-fold pairs and raw-family normalized reliance/half-L1 drift.
- `subgroup_metrics.csv`: model/holdout/category or bucket/support/four metrics/small-sample flag.

## Runtime and presentation amendment

The authoritative additional fields/formulas are in [contracts/runtime-observability.md](contracts/runtime-observability.md).

| Entity | Key fields and relationships | Invariant |
| --- | --- | --- |
| Runtime sample | sample_id, operation_id, model/config/fit/fold, workload row-ID hash, batch_size, phase, repeat, wall_ns, cpu_ns, environment_id | TRAIN-only workload; warmup vs measured vs first-load distinguished; no new fits |
| Runtime summary | exact group identity, count, min/p50/p90/p95/max/mean/SD, throughput and method | Recomputable from samples; no pooling unequal batches/fold models as one request distribution |
| Resource observation | process ID/scope, observation time, peak/start/end RSS bytes, measurement method, null reason | Lifetime high-water is not per-model memory; before/after delta is not peak allocation |
| Metric catalog entry | metric_id, name, formula, unit, direction, scope, aggregation, precision, null policy, caveat | Every typed metric has a definition; display rounding cannot influence selection |
| OOF prediction | fit/model/config/variant/fold/row IDs, actual, predicted, residual, outer-validation population | Distinct validation rows per model procedure; no holdout/reserve records |
| Performance comparison | quality winner, selected family, fastest comparable IDs, accuracy/fit-time frontier, comparability | Distinct findings; never runtime override of frozen family |
| Operational assessment | limit, scope/method, observed value, verdict, reason, evidence refs, deployment_review_eligible | Budgets optional and predeclared; missing is not passed; scientific outcome stays separate |
| Invocation event | invocation/event/operation/parent IDs, level/reason, progress, typed metrics, safe next action | Structured enum values; bounded payload; reuse does not mutate original history |
| Agent summary | identities, completeness, scientific/operational status, key typed metrics, blockers/actions/evidence refs | Consume without prose parsing, fitting or implicit approval |
| UI snapshot | schema/run/method, manifest reference, page sections, KPI/table descriptors, availability, downloads | Same-source immutable index; no live staging reader or serving activation |

`training.log` and `report.md` render existing records. `agent_summary.json` and `ui_summary.json` reference the same numeric table rows and cannot contain independently calculated competing scores. Final snapshots never inline unbounded OOF tables or event histories.

## State transitions

```text
inspect -> proposal (or blocked; no fit)
proposal + matching approval -> staged run
staged -> candidate CV -> frozen family -> conditional nested tuning
       -> paired variant CV -> final TRAIN bundles -> frozen evaluation declaration
       -> exclusive holdout-access claim -> final holdout evidence
       -> validated complete pack
```

Failures before target access leave no exposed holdout claim. Failures after claim remain recorded and cannot be automatically retried as a new evaluation. A complete identity-matched pack can be reused with zero fit/predict; a changed scientific declaration on an exposed holdout is rejected.

## Missing values and units

JSON numbers must be finite or null; CSV null is empty plus an explicit reason field when a metric is undefined. R² is null for fewer than two observations or a constant target; do not accept library force-finite substitutions. USD errors retain full numeric precision in artifacts; human reports may format dollars/percentages but selection uses raw values. Timestamps are UTC ISO 8601; durations are seconds from a monotonic clock.

Exposure, execution, fit-diagnostic, scientific-outcome and operational-budget statuses are separate fields. Combined human deployment-review eligibility is explicit and does not imply production fitness. A failed process is not an underfitting model. An educational “good-fit indication” is not the run's Good recommendation. Classification metrics are documentation with `not_applicable` status, never zero-valued model results.
