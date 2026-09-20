# Verification Evidence: Training-Only Split and Validation

**Branch**: `006-validate-training-split`  
**Implementation approval**: Explicit user message `approve`, received after review of the amended 51-task package.  
**Started**: 2026-09-19  
**Status**: Implementation in progress. No real-data model training is authorized unless the separate unseen-reserve and partition gates pass.

## Approval and checklist disposition

The approved reviewed policies include: fixed Top-2 (`job_category`, `years_of_experience`), whole-month 80/19/1 target with chronology taking precedence, five outer/three inner expanding folds, MAE/overlap/simplicity family selection, 0.20 fit-gap diagnostic, 540-fit upper bound, and at most 1,784 additional TRAIN-only benchmark prediction calls.

The requirements checklist had 32 boxes: 27 complete and 5 execution/data acceptance boxes incomplete. The user's explicit approval is treated as the instruction to proceed with implementation despite those known pending execution items. They remain pending and cannot be converted to passed by implementation alone:

- eligible unseen future inference-reserve data;
- dataset-specific partition/exposure approval;
- implementation test and real-data validation results;
- two-reader comprehension acceptance;
- final implementation approval box (now satisfied by the explicit approval and to be updated in checklist).

## T001 Approval gate

- Approved scope: offline training validation and evidence producer only.
- Not approved: UI source changes, upstream preprocessing/preparation changes, classifier training, data fabrication, automatic deployment, silent holdout reuse, or modification of inherited `uv.lock`.
- Current dataset is expected to fail unseen-reserve readiness; that is an honest Blocked result, not permission to weaken the gate.

## T002 Protected baseline identities

Captured before source implementation:

| Protected path | SHA-256 / aggregate SHA-256 |
| --- | --- |
| `src/ai_job_market/core.py` | `d435a3dcf846b5c5b778115e7ecdc7ca3e0741c27703d92b9a90bda28090f7c9` |
| `pipeline.py` | `d4cb0692ce3b028ff4d19b47aebe7c43c6f76bbc9529db6ce30d8e567ad867ee` |
| `config/project.yaml` | `da5d23253b328f2594d345509c48931f5993a3d503588c10690e95f323f116a3` |
| `src/pages/` aggregate | `c4e0b48215d7a47eb7f55370ebb954525968d284f256c8dd308b33493bb7caaa` |
| `streamlit.py` | `82e915fba0444a4335f3da45fa72a45201abf8cdb38246f9b53ef861e9f1e9d7` |
| existing `outputs/` aggregate | `20888ebc139cdb4cf3de7f6c4e2e77c87dc63c3b101b04bdb4389f5c3750113a` |
| existing `artifacts/` aggregate | `bb8f9c2dd4a31c564b01195e17e4ee138d459781652f439795d2a341fcfb58d9` |
| `requirements.txt` | `8e2f2581ea4d4ab13dc033c51c7f0654a9fee1e694c89762813464562934da4b` |
| inherited untracked `uv.lock` | `96599229fba386c9268987794da072036c3f16443f7b828004fc56c8d9b00b27` |

Only `outputs/training_validation/` and `artifacts/training_validation/` are permitted new generated namespaces. Tests must use temporary workspaces. Final verification will recompute these identities, excluding only approved new namespaces where applicable.

Project setup check: `.gitignore` already covers Python caches, `.venv`, generated outputs/artifacts, temporary files, IDE directories and Graphify output. No Docker/Node/Terraform/Helm ignore files are applicable. No ignore-file edit was needed.

## T003 Graphify and direct-source review

Command:

```text
graphify query "training validation fold summary training audit model conclusions UI evidence consumers" --budget 1200
```

Result: 278 nodes found; 34 displayed due to budget truncation. Relevant nodes included legacy `temporal_cv_splits`, `TrainingAuditSession`, `ui_evidence.py`, `ui_evidence_io.py`, `atomic_publish`, Pages 04–06 consumers, and current evidence/conclusion builders. Direct source inspection remains authoritative because traversal was truncated.

Confirmed boundaries:

- Reuse `core.candidate_models` and `core.make_model_pipeline` only.
- Do not change or relabel legacy `core.temporal_cv_splits`, tuning, ablation, finalization or primary audit contracts.
- New artifacts use their own versioned namespace and reader.
- Current Pages 04–06 remain unchanged; future UI consumption uses `ui_summary.json` only after separate approval.
- Keep the implementation additive and focused; no generic event bus or service framework.

## Test evidence log

Each behavior task records its RED and GREEN command/result below as implementation proceeds. Printed output without assertions is not acceptance evidence.

### T004–T007 — Evidence boundaries, policy and events

**RED**

```text
PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_training_evidence_v2.py
```

Failed during collection because `ai_job_market.training_evidence_io` did not exist. After initial implementation, a newly added strict feature-family test failed because prohibited `experience_level` was not rejected; this proved the policy validator needed to lock the reviewed families/search bounds.

**GREEN**

```text
9 passed in 0.02s
ruff check ... -> All checks passed
```

Evidence covers path escape/symlinks, strict run IDs, finite/null metrics, strict policy and runtime protocol, RF/family bounds, approval schema/limits, 1 MiB controls, correlated terminal events, redaction and context failure recording. Files: `tests/test_training_evidence_v2.py`, `src/ai_job_market/training_evidence_io.py`, `config/training_validation.json`.

### T008–T013 — Whole-month partitions and expanding fold evidence

**RED**

```text
PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_training_partitions_v2.py
```

Failed during collection because `ai_job_market.training_partitions` did not exist. The first implemented arithmetic test then exposed an incorrect expected cumulative-row total in the test (41 actual, not 28); the expectation was corrected from independently summed monthly counts, not by changing implementation behavior.

**GREEN**

```text
6 passed in 0.26s
```

Evidence covers deterministic whole-month proposals, missing/stale dates, exposure precedence, unequal month counts, missing calendar months, five outer folds, three nested folds, exact expansion, protected-population exclusions, monthly count reconciliation and count-tamper rejection. The static metadata fixture is `tests/fixtures/training_validation/monthly_metadata.csv`.

### T014 — Read-only inspection CLI

**RED**

```text
PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_training_validation_cli.py
```

Failed during collection because `ai_job_market.training_validation` did not exist.

**GREEN / phase checkpoint**

```text
PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_training_validation_cli.py tests/test_training_partitions_v2.py tests/test_training_evidence_v2.py
16 passed in 0.52s
ruff check affected source/tests -> All checks passed
```

`inspect` reads only fixed metadata columns and policy, reports exact counts/shares/budgets and exits Blocked for unknown/exposed reserve provenance. Tests compare the workspace tree before/after and assert zero fit/prediction counts. No root output or model artifact was generated.

### T015–T028 — Comparison, bounded search and final-evaluation primitives

RED checkpoints included missing evaluator, selection, ablation, importance, search, final-fit, holdout-score, permutation, subgroup and uncertainty functions. GREEN evidence now covers 25 paired candidate/fold records, fold-local preprocessing, strict R² null semantics, overlap/simplicity selection, six fixed ablations, RF encoded/raw-family importance and drift, the 26-slot/three-inner-fold search with exact configuration reuse, non-RF skip, fixed Top-2 matching, TRAIN-only final fits, persistent holdout access identity/locking, frozen scoring, 12-repeat permutation evidence, subgroup support flags and descriptive q90 bands.

```text
PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_training_evaluation_v2.py tests/test_training_search_v2.py
17 passed
```

T029 is now integrated behind the approval gate. The runner freezes family selection, performs conditional nested RF search only when selected, fits matched Full/Top-2 variants and all-TRAIN candidates before claiming holdout access, materializes holdout targets only after the persistent population lock, never materializes reserve targets, enforces the 540-fit ceiling, and records final diagnostics. No real-workspace holdout result is claimed.

### T030–T045 — Report, runtime and pack-contract increments

The report suite was developed RED/GREEN and now covers 5W1H, educational-only classification language, all five candidates plus final Full/Top-2 conclusions, missing evidence, Good/Bad/Blocked/Inconclusive outcomes, human/agent/UI parity, metric catalog and four authoritative-table Plotly charts. Runtime tests cover three warmups, 30 retained samples, batch truncation, wall/CPU units, linear p50/p90/p95 arithmetic, throughput, Pareto evidence, first/repeated bundle loads and operational-budget separation. Pack tests cover required inventory, strict checksums/sizes, staging publication, corruption and immutable namespace refusal. The read-only `check` CLI is implemented.

```text
PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_training_report_v2.py tests/test_training_runtime_v2.py tests/test_training_evidence_v2.py
27 passed
```

The integrated producer now publishes via same-filesystem staging, validates every required evidence checksum plus both bundle metadata/hash/size records without deserialization, and exposes read-only `check`. A 27-context benchmark integration spy proves 1,782 TRAIN-feature prediction calls with no fits; two first-loaded bundle prediction observations complete the 1,784 ceiling. Runtime evidence includes retained samples, first/repeated load timing, bundle sizes and separate scientific/operational conclusions.

### T047 — Current-workspace preflight

Command:

```text
PYTHONPATH=src .venv/bin/python -m ai_job_market.training_validation inspect --workspace . --policy config/training_validation.json
```

Result: exit `3`, `execution_status=blocked`, `reason_code=RESERVE_KNOWN_EXPOSED`, `fits_performed=0`, `predictions_performed=0`. The proposed latest reserve intersects `outputs/02_data_ready_for_ml/locked_test_raw.csv`, so no new experiment artifacts are available. T048 is deferred: no eligible future data or separately approved exact partition exists.

### T049 — Regression, lint and protected paths

```text
PYTHONPATH=src .venv/bin/python -m pytest -q
183 passed, 1 skipped in 24.93s
ruff check <all new training modules and tests>
All checks passed
git diff --check
passed
```

Recomputed protected hashes exactly match T002 for `core.py`, `pipeline.py`, `config/project.yaml`, `src/pages/`, `streamlit.py`, existing `outputs/`, existing `artifacts/`, `requirements.txt`, and inherited `uv.lock`. No training output, model bundle, preprocessing artifact or UI source was modified.

### T046 — Full fixture run, check, failure and reuse

A 15-month chronological fixture with a custodian-style exact approval was run through the real CLI in a temporary workspace. The run completed in approximately 14 seconds, remained below 540 fits, made exactly 1,784 declared benchmark/first-load prediction calls, published 5 outer plus 18 parent-labelled inner fold summaries, exported only `EVALUATION_HOLDOUT` predictions, and passed `check`. A second identical invocation returned `reused` with zero fits/predictions and byte/mtime-identical workspace files. Separate tests prove stale approval rejection before namespace creation and stage-failure cleanup before holdout access.

### T047 ratio evidence clarification

The current 1,499-row workspace proposal is TRAIN 895 (59.7065%), EVALUATION_HOLDOUT 306 (20.4136%), and INFERENCE_RESERVE 298 (19.8799%). Whole-month chronology makes the requested 80/19/1 proportions unattainable on the currently observed months, and the latest proposed reserve is already known exposed. This is an expected honest Blocked result, not scientific validation.

## Remaining acceptance gates

- T048: deferred until genuinely eligible future prepared data and exact approval exist. No real model/evidence pack was generated.
- T050: two independent human readers have not performed the comprehension walkthrough; agent tests are not a substitute.
- T051: completed below; unresolved data/human acceptance remains visible rather than being converted to passed.

## Graphify refresh checkpoint

A normal `graphify . --update --no-viz` could not process 69 changed non-code files because no supported semantic backend key is available. Following the no-key/code path, `graphify . --update --no-viz --code-only` completed: 47 code files were re-extracted, 37 were cached, and the graph now contains 1,490 nodes, 2,827 edges and 81 communities. Documentation semantics therefore remain outside this refresh and that limitation is not hidden.

## T051 — Final review and synchronization

Final code-only Graphify refresh re-extracted 9 changed code files (75 cached) and produced 1,515 nodes, 3,006 edges and 86 communities. The four expected non-AST JSON/TOML inputs remain zero-node warnings. A fresh-context five-axis review checked correctness, readability, architecture, security boundaries and bounded performance. Required fixes applied during review included: offline-inline Plotly assets, generated-path symlink containment, artifact bundle checksum/metadata validation without deserialization, honest non-circular UI manifest reference, producer/dependency identities in run reuse, TRAIN-only target materialization before the lock, first-loaded prediction evidence, all parent-labelled inner-fold summaries, and final-estimator encoded importance.

Final verification:

```text
PYTHONPATH=src .venv/bin/python -m pytest -q
183 passed, 1 skipped in 25.18s
ruff check src/ai_job_market/training_*.py tests/test_training_*_v2.py tests/test_training_validation_cli.py
All checks passed
git diff --check
passed
```

All T002 protected hashes still match exactly. `spec.md`, `plan.md`, `quickstart.md`, both contracts, checklist, tasks and this verification record now reflect implemented/fixture-verified status. T048 (eligible real data) and T050 (two independent human readers) remain explicitly pending. No UI, preparation, legacy output, existing artifact, dependency, serving pointer or inherited `uv.lock` content changed.
