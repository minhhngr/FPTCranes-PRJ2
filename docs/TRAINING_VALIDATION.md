# Training-validation evidence

This feature is an **offline, training-only** workflow. It does not change preprocessing, prepared source data, deployment behavior, or Streamlit pages. Evidence is written only under an immutable run namespace:

```text
outputs/training_validation/tv-<32 lowercase hex>/
```

## Safe preflight

Run the read-only inspection from the repository root:

```bash
.venv/bin/python training_validation.py inspect \
  --workspace . \
  --policy config/training_validation.json
```

The command reads only cleaned date metadata, proposes whole-month boundaries, checks known local exposure evidence, and prints JSON. It performs **zero fits and zero predictions**. Exit code `3` means execution is blocked. The currently committed dataset is expected to be blocked because it cannot establish a genuinely unseen future inference reserve.

A compliant execution additionally requires a separately reviewed approval control containing the exact source hash, policy hash, proposal hash, partition hash, boundaries, row counts/shares, reviewer, timestamp, and an eligible reserve exposure attestation. `build_approval_template()` may generate the hash/count material for review, but it does not attest exposure or approve a run. A changed source or boundary invalidates approval; changing a run ID does not bypass the holdout lock.

After the custodian has reviewed and completed a workspace-local approval file, run and validate the immutable pack with:

```bash
.venv/bin/python training_validation.py run \
  --workspace . \
  --approval approval.json \
  --policy config/training_validation.json

.venv/bin/python training_validation.py check \
  --workspace . \
  --run-id tv-0123456789abcdef0123456789abcdef
```

`run` fails closed on stale hashes, known/unknown reserve exposure, conflicting namespaces, or an existing non-reusable holdout claim. An exact complete run is returned as `reused` before fitting, predicting, loading a bundle, or writing history. The current committed data is **not** eligible for this command.

## Split and validation method

- `TRAIN` is approximately 80%, subject to whole-month chronology.
- `EVALUATION_HOLDOUT` is approximately 19%, strictly later, opened once only after model choice and final fitting.
- `INFERENCE_RESERVE` is approximately 1%, strictly latest, and is never scored.
- Five outer folds use expanding monthly history inside `TRAIN` only.
- If Random Forest is selected, each of five outer contexts plus final TRAIN uses three expanding inner folds and `GridSearchCV` over the declared `n_estimators` × `max_depth` grid. `min_samples_leaf` and `max_features` are fixed and recorded for every candidate.
- The search scores negative MAE (reported as lower-is-better MAE), records mean/std R² as secondary descriptive evidence, candidate ranks, timings, fold IDs, parameters, and status. The producer writes one raw structured event per GridSearchCV context and candidate result.
- Candidate selection uses unrounded mean temporal-CV MAE and the declared overlap/simplicity rule. Runtime evidence does not override scientific selection.

## Reading a complete pack

Start with these files:

1. `manifest.json` — schema, `gridsearchcv-temporal/v1` method identity, run identity, inputs, hashes, and file inventory. Older manual-search packs are invalid for revised-method claims.
2. `report.md` — human-readable 5W1H narrative, fold method, findings, limitations, and next actions.
3. `fold_explanation.md`, `fold_summary.csv`, `monthly_row_counts.csv` — exact monthly construction and row counts.
4. `candidate_summary.csv`, `family_selection.json`, `model_conclusions.json` — five-model comparison and explicit decisions.
5. `holdout_metrics.csv`, `holdout_predictions.csv` — one-time evaluation evidence; never reserve results.
6. `runtime_summary.csv`, `operational_assessment.json` — descriptive measurements and separately configured budget verdicts.
7. `agent_summary.json` and `ui_summary.json` — machine-readable projections derived from the same authoritative tables.
8. `events.jsonl` and `training.log` — append-only structured lifecycle events and a detailed English evidence-derived execution log. The readable log answers 5W1H, reconciles requested versus actual whole-month shares, lists all folds and 25 candidate-fold metrics, records fit diagnoses, ablation/importance/drift, every GridSearchCV candidate or skip reason, Full/Top-2 and holdout evidence, residuals, subgroup/q90 evidence, and the final scientific/operational conclusion.

Missing, failed, or skipped evidence must remain explicit. It must not be converted into zeros, favorable claims, or fabricated charts.

### Classification metrics are educational only

Classification metrics are not applicable because the target is continuous annual salary. For interpretation only: a confusion matrix counts predicted versus actual classes; accuracy is the overall correct fraction; precision measures correctness among predicted positives; recall measures coverage of actual positives; F1 is the harmonic mean of precision and recall; and support is the number of examples in each class. Micro averaging pools decisions, macro averaging weights classes equally, and weighted averaging weights by support. ROC-AUC and PR-AUC require ranking/probability scores and declared positive classes; PR-AUC is often more informative under severe imbalance. Undefined denominators require explicit null reasons. This workflow creates no classes, class probabilities, confusion matrix, or artificial classification scores.

## Validation and limits

A complete pack can be checked without deserializing model bundles:

```python
from ai_job_market.training_evidence_io import validate_complete_pack

manifest = validate_complete_pack(".", "tv-0123456789abcdef0123456789abcdef")
```

This verifies the schema, complete status, required inventory, containment, evidence and bundle sizes, and SHA-256 checksums without deserializing a model. It does not prove that externally exposed data remained unseen, that the model is fit for deployment, or that an operational budget passed. Those limits are explicit in the report and conclusion evidence.
