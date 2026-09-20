# Contract: Supplemental UI Evidence v1

**Status**: Proposed for task approval; not an existing implemented API.  
**Owner**: `src/ai_job_market/ui_evidence_io.py`  
**Producer**: `ui_evidence.py` / `ui_evidence_training.py`  
**Consumers**: `src/pages/model_evidence.py`, Pages 04–06, `salary_inference.py`, release-contract tests.

## 1. Command surface

```text
PYTHONPATH=src .venv/bin/python -m ai_job_market.ui_evidence --workspace PATH [--check]
```

- `--workspace`: required local existing directory holding baseline `outputs/` and `artifacts/`; resolve it once. Checkout code/config/dependency identities come from the running project, not arbitrary Python files inside the workspace.
- Normal mode: validate inputs, reuse matching components, generate only missing/invalidated components, validate and atomically publish the supplemental manifest/current pointer. No unconditional `--force` retraining flag.
- `--check`: resolve/validate existing evidence and source dependencies only; no fitting, prediction, output creation or pointer updates.
- Exit codes: `0` required components complete and valid; `2` invalid input/contract or absent required evidence in check mode; `1` generation/runtime failure. Optional unavailable tuning history may still yield `0`, with a visible reason recorded.
- Console summary names reused/generated/unavailable components and resolved evidence ID, with no invented success for failed steps. Exceptions preserve the original current pointer and include actionable file/command context.
- No UI button runs this command. No change to `pipeline.py` or `core.py` is required.

## 2. Paths and publication

```text
outputs/ui_evidence/current.json
outputs/ui_evidence/<evidence_id>/
  manifest.json
  evaluation_declaration.json
  candidate_fold_metrics.csv
  candidate_summary.csv
  candidate_test_metrics.csv
  fold_membership.csv
  rf_fold_importance.csv
  rf_importance_drift.csv
  variant_fold_metrics.csv
  variant_metrics.csv
  variant_test_predictions.csv
  variant_encoded_importance.csv
  variant_permutation_importance.csv
  scenario_policy.json
  benchmark_examples.csv
  tuning/                            # validated snapshots, optional
  charts/                            # generated HTML from authoritative tables
artifacts/ui_evidence/<evidence_id>/
  top2_model.joblib
  top2_metadata.json
```

Use a generated alphanumeric/hyphen/underscore evidence ID (UTC timestamp plus random suffix is sufficient); users do not enter it as a path. The pointer has `schema_version`, `evidence_id`, project-relative `manifest_path` and `manifest_sha256`. Payloads are immutable after publication. New manifests may reference hash-verified immutable files from an earlier supplemental run to avoid refitting unchanged components; never use symlinks.

Create staging directories exclusively within the namespace, write/flush/close payloads, validate referenced files and model reload, then atomically replace the pointer last. If publication fails, leave the prior pointer usable; incomplete staging is not discoverable as a valid run. Refuse destination symlinks, `..`, absolute manifest payload paths, escapes outside the resolved workspace and paths outside the allowlisted baseline/supplemental subdirectories. Hash verification precedes trusted-local joblib loading. This is integrity protection for local artifacts, not permission to load an untrusted uploaded pickle.

## 3. Required manifest fields

| Field | Meaning / validation |
| --- | --- |
| `schema_version` | Integer `1`; reject unsupported versions |
| `evidence_id`, `created_at_utc` | Unique run identity and ISO timestamp |
| `source_run_id` | Baseline metadata run ID if present; never fabricate one |
| `target`, `full_features`, `top2_features` | `annual_salary_usd`; verified full list; exactly `[job_category, years_of_experience]` in order |
| `source_files` | Workspace-relative path, SHA-256, size, role for raw/prepared data, split summary, baseline metadata/contract/model and imported outputs |
| `producer` | Checkout-relative producing files and hashes, contract version, config/lock hashes, Python/library versions and seed |
| `datasets` | DEV/test snapshot hashes, actual counts, period extents, ordered record-ID policy and disjoint membership identity |
| `models` | Variant/model IDs, estimator class/parameters, feature order, configuration digest, bundle reference/hash when serving, original selection identity |
| `components` | Named dependency fingerprint, `generated`/`reused`/`unavailable`, references, reason, historical limitations |
| `files` | Allowlisted path, SHA-256, media/schema identity, required columns and row count for each payload |
| `limitations` | Historical test exposure, inherited tuning/selection policy, synthetic-looking data and empirical-band limitations |

Required generated/verified components for a complete real release: `comparison`, `variants`, `policy`, `benchmarks`, `serving`, `charts`. `tuning_history` may be unavailable with a reason; its absence cannot disable valid prediction. Benchmarks may have fewer than three rows if strict eligible rows are scarce; the count and reason are explicit.

A manifest checksum alone is not historical experiment provenance. Imported legacy files retain `origin=legacy`, actual source hash, model/partition correspondence and known limitations. Unsupported legacy Train_* columns are excluded from accepted evidence. The new evaluator supplies paired metrics instead.

## 4. Component dependencies and reuse

| Component | Dependencies that invalidate it | Required response |
| --- | --- | --- |
| Comparison fold scores/drift | DEV contents/order, frozen candidate params, features/folds/seed, training helper/producer code, runtime/lock | Regenerate affected fold evaluations; no mixing old validation/new train scores |
| Candidate test diagnostics | Frozen candidate configs/fitted DEV dependency, historical test snapshot, metric code | Recompute required predictions/metrics; reuse immutable matching results if complete |
| Full/Top-2 variant CV and Top-2 fit | DEV/folds, fixed full params and full feature contract, Top-2 feature list, seed, producing code/lock | Regenerate affected variant components; never automatically refit primary full bundle |
| Historical predictions/metrics/importance | Test snapshot, verified bundle/configuration, scoring/importance code and seed | Recompute affected evidence; no new tuning/promotion |
| Policy | DEV input fields, metadata enum identity, policy version/code | Rebuild target-independent policy; no estimator fit |
| Benchmarks | Policy, historical test row identity, variant prediction identity, selection rule | Reselect deterministically and attach predictions; no extra fitting |
| Display/export/manifest schema | Contract and formatting code, source table identities | Rewrap/revalidate or mark incompatible; do not fit estimators unless a producing model dependency changed |

Fingerprint only actual dependencies; a chart-caption edit is not a training change. If computation can be avoided only by inventing an unrecorded identity, do not reuse. A complete unchanged normal invocation must make zero fit calls and leave baseline artifacts unchanged.

## 5. Data identity and evaluation declaration

Prepared partitions lack original `job_id`. Use `record_id = <partition>:<dataset_sha256>:<zero-based-row-offset>`; hash exact snapshot bytes and preserve original row order. This is an evidence-row identity, not a real job identifier. Duplicate row contents remain distinguishable. Sort benchmark candidates by numeric row offset, not salary/error/hash magnitude.

`evaluation_declaration.json` records target, feature lists, candidate/final settings, seed, DEV/test partition identities, exact fold train/validation IDs, source identities and `selection_policy=frozen_diagnostic_only`. Write it before reading test targets for the new evaluation. Target mutation tests must not change configurations, role policy or chosen benchmark source-row offsets. Changing target bytes legitimately changes the snapshot hash and thus record IDs; compare selection offsets, not cross-snapshot IDs. Test scoring cannot affect Top-2 settings or the existing full-model selection.

## 6. Table schemas and calculations

All CSVs use UTF-8, unique names, declared types, no unnamed index column. Nonfinite JSON values are forbidden; unavailable metric values use null/blank plus a reason rather than `NaN`/`Infinity` strings. Join by identities, never assumed display names or positional reordering across different snapshots.

### Shared identities

Every metric table carries its producing `evidence_id`, `model_id`, `configuration_id`, `feature_variant`, `partition`. Reused components retain that original producing ID; the active manifest explicitly links and validates their identical dependency fingerprints rather than rewriting immutable rows to a new ID. Model IDs distinguish `candidate:<family>`, `full:selected`, `top2:fixed`. Tables never combine different configurations merely because each is called Random Forest.

### Fold tables (`candidate_fold_metrics.csv`, `variant_fold_metrics.csv`)

One row per model/configuration/fold: shared identities; `fold_id`, `train_period`, `validation_period`, `train_rows`, `validation_rows`, `encoded_feature_count`, `fit_time_s`, `train_predict_time_s`, `validation_predict_time_s`, plus `train_MAE`, `train_RMSE`, `train_R2`, `train_MedAE`, `validation_MAE`, `validation_RMSE`, `validation_R2`, `validation_MedAE` and optional undefined-score reason. Train/validation values come from the same fitted fold model. `fold_membership.csv` contains `fold_id, partition_role, record_id` under the declared DEV snapshot.

Summary mean is an unweighted mean of fold scores, matching the existing evaluator; MAE SD uses `ddof=0`. Do not compute pooled CV metrics and call them fold means. Fit/predict times are nonnegative measured durations, not source-document literals.

### Candidate summary/test

`candidate_summary.csv`: shared identities; family, effective fold count, each train/validation metric mean, validation MAE SD, mean fit/predict times and recorded candidate parameters/reference. Rank is a view derivation by validation MAE ascending, ties displayed equally (stable model-ID order only for layout), not a new full-model promotion decision.

`candidate_test_metrics.csv`: shared identities; test dataset ID/count/period, MAE/RMSE/R²/MedAE, frozen settings, `historically_exposed=true`, `selection_use=none`. No UI uses test metrics to rerank the family winner.

### Variant metrics/predictions

`variant_metrics.csv`: shared identities and `evaluation=dev_cv_mean|historical_test`, dataset/fold identity, row count, MAE/RMSE/R²/MedAE. Historical test entries additionally include `q90_abs_error_usd`, `q90_basis=historical_test_absolute_errors`, `q90_quantile=0.9`, `q90_method=linear`, observed `coverage` and actual denominator. Top-2 and full must share test identities for side-by-side comparison.

`variant_test_predictions.csv`: one row per `(model_id, record_id)` with actual salary, prediction, `residual_usd=actual-prediction`, absolute error and retained role/profile columns needed for diagnostics. All-DEV Top-2 fit and saved full model are used, not fold models. Verify saved full outputs against corresponding baseline predictions/metrics before accepting that relationship; conflict disables the full component, not an automatic replacement fit.

### Importance

`rf_fold_importance.csv`: `model_id, configuration_id, fold_id, encoded_feature, raw_family, importance`; map encoded names using transformer output metadata, not fragile underscore splitting. For across-fold comparisons, absent encoded features contribute zero to the union, with the rule disclosed.

`rf_importance_drift.csv`: identity, `scope=encoded|raw_family`, feature, mean, population SD, min/max, fold count and CV=SD/mean (unavailable if mean zero). Sum each family's importances within each fold first before computing family drift.

`variant_encoded_importance.csv`: model/configuration, encoded feature, raw family, importance. `variant_permutation_importance.csv`: model/configuration, raw feature, `mae_increase_usd`, `std_usd`, repeats=12, seed and historical test identity. Negative importances are valid evidence; don't clip them.

### Scenario policy and examples

Policy schema is in [scenario-ui.md](scenario-ui.md). `benchmark_examples.csv` includes record/partition identity, source row offset, title/category/experience, known original attributes, actual salary, full/Top-2 predictions/errors, `included_in_benchmark=true`, `historically_exposed=true`, `pristine=false`, selection rule and strict-policy eligibility. Generate actual example labels from these rows; never hardcode stakeholder sample profiles.

### Tuning and charts

Validated optional tuning snapshots retain actual columns/parameters and stage identity. Each stage includes provenance and `methodology=inherited_non_nested_dev_search`; use actual `CV_MAE`, `CV_R2`, `CV_RMSE`, `CV_MedAE` where present. Final applied values come from the verified bundle/metadata. Contradictory fixed captions or rationale strings are not scientific evidence.

Generated HTML figure files reference the same table/model identities as interactive views. They are supplemental outputs, not authoritative measurements or manually illustrated source-document charts.

## 7. Unavailability and isolation

- Missing current pack: page-specific actionable error with the offline command; never auto-generate.
- Invalid required file/schema/hash/model/policy: block its dependent section or serving. An invalid manifest/pointer blocks the entire supplemental pack.
- Optional unavailable tuning history: warning in that section while valid comparison/prediction works.
- Baseline/supplemental mismatch: visible incompatible-evidence state, no implicit fallback to similarly named legacy files.
- Workspace switch: resolve that workspace's pointer; never reuse the baseline workspace's cached bundle or benchmark rows.
- `core.py`, `pipeline.py`, old output packs and old model files must retain pre-change hashes during new generation. Integration tests assert this and force generation failures before pointer replacement.
