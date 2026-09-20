# Contract: Offline Training Validation v1

**Status**: Implemented and fixture-verified; real-data execution remains blocked by unseen-reserve and partition-approval gates.  
**Owner**: Offline training only. Existing UI/primary-pipeline contracts are unchanged.

**Companion contract**: [runtime-observability.md](runtime-observability.md) defines required runtime samples, extra regression metrics, human/agent views, typed UI snapshots and operational readiness. Both documents form the isolated `training-validation/v1` contract; it is not activated by the existing UI or serving pipeline.

## CLI interface

```text
python -m ai_job_market.training_validation inspect --workspace ROOT [--policy FILE]
python -m ai_job_market.training_validation run --workspace ROOT --approval FILE [--policy FILE]
python -m ai_job_market.training_validation check --workspace ROOT --run-id ID
```

- `inspect`: no fitting, prediction, target-aware analysis or writes; print a machine-readable JSON proposal on stdout and English explanation on stderr. Output includes readiness, proposed boundaries/counts/shares/deviations, input/policy/proposal hashes, fold budget and exposure blockers. No claim the proposed allocation is already approved.
- `run`: explicit approval required. Reject incompatible inputs before fits. Generate only new training-validation outputs; no preprocessing, shell subprocess pipeline, UI mutation or serving activation. Reuse a complete matching run before any fitting or holdout access.
- `check`: read-only schema/hash/provenance validation of a complete pack; no fitting, prediction, artifact generation or joblib deserialization. Bundle integrity is checked by hash/metadata; only the trusted offline producer loads its own local models when needed.
- Exit 0 success (including complete reuse); 2 invalid arguments/schema/path; 3 scientific/data readiness blocked; 4 runtime/publication failure. JSON errors contain `code`, English `message`, `field_or_artifact`, `next_action`; stderr may carry context but no secrets or untrusted raw content as executable instructions.

## Input contract and validation

`ROOT` is an existing local workspace. Code/policy/version identities come from the checkout. Raw input is fixed at `ROOT/outputs/01_data_basic_clean/basic_clean.csv`; no arbitrary model/input pickle paths. Configuration and approval paths must resolve within ROOT or the checkout as documented, have real non-symlink ancestors and be regular JSON files. All generated destinations stay under the two new namespace roots. Reject traversal, symlinks, existing incompatible run directories, unknown JSON fields and invalid selectors before writes.

Run IDs are generated `tv-` + 32 lowercase hexadecimal characters derived from the full experiment fingerprint; manifests retain the full digest and reject any shortened-ID collision. User-supplied IDs for check must match `^tv-[0-9a-f]{32}$`. File sizes: policy/approval/exposure JSON <=1 MiB each; cleaned CSV <=100 MiB and <=100,000 rows (resource guard, not a claim of estimator scalability). Input dates require integer year 1900–2100, month 1–12; target and numerical model values finite where authorized for access; years of experience nonnegative. Required raw columns/strings must match the existing preprocessing contract; no new cleaning/imputation. Ambiguous duplicate non-target fingerprints block an unseen assertion until resolved by provenance, not by dropping rows.

Policy keys: `schema_version`, seed (integer 0..2^32-1), feature_policy reference/hash, fixed fold counts (5 outer/3 inner), split targets (0.80/0.19/0.01), six feature-family definitions, fixed two-feature list, bounded RF stage arrays, `n_jobs=1`, permutation repeats=12, diagnostic gap=0.20 and small-subgroup threshold=20. Add a `runtime_protocol` object with method version, warmups=3, measured_calls=30, batch_sizes=[1,100] capped/deduplicated by actual TRAIN support, max_benchmark_predict_calls=1784, numerical_threads=1 and sequential execution. Add console verbosity enum concise/verbose (presentation only, no event suppression). Default scientific-method values match `research.md` and the runtime companion contract. Changes require a new policy identity and review, not a hidden CLI override. Unknown feature names, blocked fields, duplicate features, expanded search bounds or nonfinite values fail closed.

Approval keys: source hash, policy hash, train_end and holdout_end as YYYY-MM, actual counts/shares, proposal/partition hash, reviewer, UTC approval time, exposure declarations and optional positive finite `max_holdout_mae_usd`/`max_holdout_rmse_usd`. Limits must both be present for a possible scientific Good status. Optional positive finite operational limits `max_run_wall_s`, `max_final_batch1_p95_ms` and `max_bundle_bytes` are separately scoped and assessed as specified by the companion contract; missing limits do not pass by default. The reserve is every observed month after holdout_end; no reserve row sampling. All source rows belong to exactly one population, with no silently excluded month. Validate lineage against known historical packs and local exposure ledger. An assertion cannot override known prior use; unknown/exposed reserve is blocked. Actual count/share mismatch rejects stale approval.

## Files and producer/consumer mapping

```text
outputs/training_validation/<run_id>/
  manifest.json
  partition_declaration.json
  partition_membership.csv
  fold_membership.csv
  fold_summary.csv
  monthly_row_counts.csv
  fold_explanation.md
  model_conclusions.json
  events.jsonl
  report.md
  training.log
  metric_catalog.json
  agent_summary.json
  ui_summary.json
  oof_predictions.csv
  runtime_samples.csv
  runtime_summary.csv
  performance_comparison.csv
  accuracy_runtime_tradeoff.csv
  operational_assessment.json
  candidate_fold_metrics.csv
  candidate_summary.csv
  runtime.csv
  fit_diagnostics.csv
  ablation_metrics.csv
  fold_importance.csv
  importance_drift.csv
  family_selection.json
  tuning_trials.csv                    # or manifest skipped reason
  tuning_summary.json
  variant_fold_metrics.csv
  evaluation_declaration.json
  holdout_metrics.csv
  holdout_predictions.csv
  encoded_importance.csv
  permutation_importance.csv
  subgroup_metrics.csv
  uncertainty.json
  conclusion.json
  charts/model_comparison.html
  charts/accuracy_runtime_tradeoff.html
  charts/holdout_diagnostics.html
  charts/feature_importance.html
artifacts/training_validation/<run_id>/
  full.joblib
  full.metadata.json
  top2.joblib
  top2.metadata.json
outputs/training_validation/holdout_access/<holdout_id>/
  access.json
```

- Page 04 future consumer: candidate folds/summary/runtime/fit diagnostics, mandatory fold-method walkthrough with monthly/exact fold row counts, ablation, RF importance/drift, family-selection conclusion and individual conclusions for all five candidate models.
- Page 05 future consumer: tuning with parent-labelled inner-fold row-count summaries, full/top2 fold metrics, final holdout metrics/predictions, importance, subgroup errors, uncertainty and separate full/Top-2 conclusions plus overall recommendation. Companion contract sections 9–10 define these required summaries and their source reconciliation.
- Page 06: no new consumer or model activation; reserve membership is recorded for a later approved integration only. No reserve feature/target/prediction export in this feature.
- A reader must bind every displayed claim to run, model/configuration, feature variant and partition. It must not merge new metrics with old UI bundles or treat the final holdout as CV.
- The immediate consumer is the new offline report/read-only checker. Contract tests must load the produced pack using that same validator; UI source changes are not needed.

`manifest.json` contains schema/method version, full input/producer/config/dependency fingerprints, Python/package/hardware metadata, seed, target/feature schema, requested and actual ratios, exposure declarations, fit budget/actual/reused counts, selection/declaration IDs, bundle hashes, section states, scientific outcome, and every output path/SHA-256/row count. No absolute external paths in exported evidence references. A checksum proves identity, not historical methodology or unseen-data authenticity.

## Event contract and 5W1H

Each JSONL event has schema_version, run_id, invocation_id, event_id, operation_id, parent_operation_id, monotonic sequence, UTC timestamp, level, event_type, reason_code, stage_id, operation, status, English message, parent/fold/trial/model/configuration IDs when applicable, elapsed_s, counts/progress, typed metrics, safe next_action, evidence references and bounded structured details. Exact field rules, caps and human/agent rendering are specified in the companion contract. Required lifecycle events include run/stage/fold/trial started, completed, failed, cancelled and reused; selection frozen; evaluation locked; publication complete/failed. Every started unit has a terminal event where process control permits; a crash leaves an explicit incomplete run on subsequent inspection, never assumed completion.

Run context answers:

| Question | Required answer |
| --- | --- |
| Who | Local operator/custodian/reviewer identifiers supplied for audit; no credentials |
| What | Target, candidate families, feature policy, experiment and output meaning |
| When | Run time, data periods, fold chronology, approval/freeze/access times |
| Where | Contained input/evidence paths and immutable run/partition/model IDs |
| Why | Training purpose, why temporal validation, selection rules, limitations |
| How | Whole-month split, fold-local preprocessing, fit/score/tune steps and budget |

Log configuration/metrics/fold membership and decisions, not every raw record value or internal tree node. Full numeric evidence lives in linked tables, so the English log remains readable while the audit is complete.

## Metrics and interpretation

MAE = mean absolute residual, RMSE = square root of mean squared residual, MedAE = median absolute residual (USD/year; lower better). R² = 1 - SSE/SST (unitless; higher better), null with reason if undefined. Report paired TRAIN/outer-validation metrics, five-fold mean/population SD and separately labelled pooled out-of-fold values. Do not average R² across rows or use rounded report values for ranking.

Tuned outer-fold metrics assess the conditional tuning procedure, not the all-TRAIN final estimator and not an unbiased entire family-selection pipeline. Factory candidate metrics remain separate. Holdout rankings never replace the frozen winner. Importance and interval formulas are specified in `research.md`; no reserve metrics exist. Permutation scores may be negative and encoded coefficient magnitudes are not labelled impurity.

Binary/multiclass glossary explains confusion matrix, accuracy, precision, recall, F1, support, micro/macro/weighted averaging, probability/ranking requirements for ROC-AUC and PR-AUC, class imbalance and undefined denominators. State `not_applicable: continuous salary regression` for this experiment. No classification tables, labels or artificial scores.

## Publication, reuse and one-time evaluation

Use exclusive staging directories on the same filesystem, validate pack/bundle hashes and tables, then atomically rename completed outputs. If separate evidence/model renames cannot be a single transaction, manifest visibility is the final commit point: readers accept only the final complete evidence manifest referencing already verified model files. No existing UI `current.json` is changed. Failed/partial runs remain clearly non-consumable.

Reuse key includes prepared data, approved memberships/exposure, full policy, actual estimator parameters, producer/helper source hashes, schema/consumer contract, Python/dependency lock identity and package versions. Preserve inherited untracked lock as-is and record its content identity; missing required identity blocks reproducibility approval rather than modifying dependencies. A real policy change invalidates old *new-experiment* evidence; legacy packs stay historical and untouched.

Holdout access is population-keyed, not run-ID-keyed. Atomic claim before target access guards concurrent runs. Verify ledger source membership against known fingerprints across repackaged inputs. No fresh evaluation with a changed declaration on an already consumed holdout, and no reset on crash. Returning existing verified metrics is permitted; generating new metrics from the same exposed population is not a fresh final test. A local ledger cannot police external copies or deleted history; disclose this trust boundary.

## Outcome and handoff

`conclusion.json`: execution_status, scientific_outcome, criteria_evaluated, failed/missing criteria, selected_model_id, evidence refs, limitations and next_action. Good/Bad/Blocked/Inconclusive scientific semantics match `plan.md`; operational budget/readiness and combined deployment-review eligibility are separate fields defined in the companion contract. Empirical success alone is not a production guarantee. Suggested lifecycle is feature review → training/CV → conditional tuning → frozen evaluation → human deployment review → separately collected monitoring evidence → newly approved retraining. No auto-loop against a consumed holdout, no deployment or monitoring infrastructure.
