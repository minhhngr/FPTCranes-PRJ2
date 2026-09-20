# Verification: Show Offline Training Validation on Pages 04–05

**Branch**: `007-show-training-validation`  
**Approved**: User explicitly said “approve tasks and implement” after mandatory transcript/log amendments.  
**Started**: 2026-09-20  
**Status**: Implementation in progress. No real training or artifact regeneration is authorized.

## T001 — Approved boundary

Approved behavior:

- read-only, collapsed Page 04/05 expanders;
- latest valid immutable training-validation pack;
- truthful unavailable state with no fallback;
- exact outer/inner fold row/month explanation;
- all 25 candidate-fold steps and per-model conclusions;
- tuning/variant/final evidence transcript;
- raw `training.log`/`events.jsonl` clearly separated from the evidence-derived transcript;
- original log/report/summary downloads plus complete generated transcript CSV.

Not approved: pipeline edits, producer/schema changes, real training, artifact regeneration, Page 06, model activation, preprocessing/source changes, deployment or fabricated evidence.

## T002 — Protected baseline

| Path | SHA-256 / aggregate |
| --- | --- |
| `src/ai_job_market/training_evidence_io.py` | `76d713cadbaff14136327d388f4a0138440c22cc91745c54de8e135cfca126d6` |
| `src/ai_job_market/training_validation.py` | `eef19641f15cc14aa1ba9513bbfcb5a027531262e4ae288c5033fb19e3931382` |
| `pipeline.py` | `d4cb0692ce3b028ff4d19b47aebe7c43c6f76bbc9529db6ce30d8e567ad867ee` |
| `src/pages/page06_prediction.py` | `867e874688b258f4f6aab134d46557dccdc9d0666f0d2ed69a18f21a1e1b2b04` |
| `src/ai_job_market/ui_evidence_io.py` | `f77fb675719065cbf3df0bb82b12d7d6db561d7398afd59c283cc35a17157b45` |
| `config/project.yaml` | `da5d23253b328f2594d345509c48931f5993a3d503588c10690e95f323f116a3` |
| `requirements.txt` | `8e2f2581ea4d4ab13dc033c51c7f0654a9fee1e694c89762813464562934da4b` |
| inherited `uv.lock` | `96599229fba386c9268987794da072036c3f16443f7b828004fc56c8d9b00b27` |
| existing `outputs/` aggregate | `20888ebc139cdb4cf3de7f6c4e2e77c87dc63c3b101b04bdb4389f5c3750113a` |
| existing `artifacts/` aggregate | `bb8f9c2dd4a31c564b01195e17e4ee138d459781652f439795d2a341fcfb58d9` |

The dirty tree inherited the approved feature-004 implementation and untracked `uv.lock`; preserve all of it.

## T003 — Dependency review

Graphify query:

```text
graphify query "Page 04 Page 05 evidence loaders expanders unavailable state training validation ui_summary" --budget 1600
```

It found the narrow integration path through `page04_model_comparison.py`, `page05_best_model.py`, `model_evidence.py`, `training_evidence_io.validate_complete_pack()`, and the existing AppTest/view suites. Direct inspection confirms the renderer must run after page title/caption and before the existing supplemental-evidence early return. The simplest design is one new consumer-only module and two small calls; no generic plugin, pointer, pipeline or producer edit.

## RED/GREEN and acceptance log

- RED: `tests/test_training_validation_views.py` initially failed collection because `pages.training_validation_presentation` did not exist.
- GREEN: discovery, strict validation, projections, transcript generation and both page renderers pass against complete, missing, staging-only, malformed-ID, symlink, corrupt-newer, skipped-tuning, unavailable-optional and reserve-contaminated fixtures.
- A temporary synthetic producer pack was loaded once to verify the consumer against the actual producer schema: available, five outer folds, 18 inner folds, skipped tuning accepted and complete transcript projected. It was created outside the repository and deleted automatically; no real data or repository artifact was trained/regenerated.
- Real-entrypoint AppTest proves the current workspace shows collapsed `Training validation unavailable` content on Page 04 and both Page 05 expanders while existing supplemental charts continue to render.
- Valid-pack page AppTests prove all expected fold/candidate/tuning/holdout tables and original/generated downloads render before the independent supplemental-source error. Operation spies cover training comparison, final fitting, scoring, benchmarking, publication, workspace execution and `joblib.load`; a before/after file-hash snapshot proves page rendering writes nothing.
- Source parity: five outer, 18 inner, 25 candidate-fold, six tuning-context plus all trial, ten variant-fold, two final-fit, seven holdout, seven residual-summary and seven conclusion transcript rows reconcile to authoritative source rows. Original log/event/report/summary bytes and hashes match the pack; the transcript download is complete and deterministic.
- Optional R², runtime and estimator-importance absence remains local; skipped tuning reasons and small-subgroup flags remain visible without favorable defaults.

## Automated verification

- Focused pure/AppTest and existing model UI suites: **41 passed** before final contract-label refinements.
- Final focused Page 04/05 pure/AppTest suite: **26 passed**; expanded evidence-contract focus: **44 passed**.
- Full suite: **198 passed, 1 skipped**.
- Ruff on changed Python: passed.
- `git diff --check`: passed.
- Actual producer-schema temporary projection: passed (`available=True`; transcript complete).
- Final Graphify incremental refresh after the trust overview: completed; 1,561 nodes, 3,106 edges, 100 communities. It warned that four JSON/config sources produced zero nodes, but code extraction completed.

## Protected-boundary comparison

All T002 hashes are unchanged, including producer modules, pipeline, Page 06, config, requirements, inherited `uv.lock`, and existing `outputs/`/`artifacts/` aggregates. `outputs/training_validation/` remains absent in the repository workspace. No pointer, dependency, model bundle or generated training evidence changed.

## Additive third-party trust overview approval

The user explicitly approved an additive visible training/validation summary while requiring that the current Page 04 and Page 05 layouts remain otherwise unchanged, and noted that the summary may be removed after verification. The implementation therefore adds one isolated native Streamlit container before the existing collapsed details. Valid packs show verified evidence counts and run/trust identity; unavailable workspaces show capability scope while explicitly denying a completed/trusted run and displaying no fallback metrics. Existing charts, tabs, conclusions, routes, and supplemental loaders were not edited.

Failing-first AppTest assertions were added for the visible heading, valid-pack status/counts, and unavailable capability boundary. The focused Page 04/05 suite passes with all existing tests and operation/write spies intact. Final regression after this addition and the CLI reliability fix: **199 passed, 1 skipped**; Ruff and `git diff --check` passed; protected hashes remained unchanged and the root training-validation pack remains absent.

## Reported CLI import failure

Reproduction showed that `-m ai_job_market.training_validation` requires `PYTHONPATH=src` in a source checkout; the user's later invocations omitted that environment assignment and failed before inspection. A failing subprocess regression test captured the issue. The repository now provides `training_validation.py`, a narrow source-checkout wrapper, and the UI/docs display this exact command:

```bash
.venv/bin/python training_validation.py inspect --workspace . --policy config/training_validation.json
```

The command was executed from the repository root with `PYTHONPATH` absent. It imports successfully and exits 3 with structured `execution_status=blocked`, `reason_code=RESERVE_KNOWN_EXPOSED`, `fits_performed=0`, and `predictions_performed=0`. This is the scientifically correct current-workspace result; it cannot produce a trustworthy complete pack because the proposed reserve is historically exposed.

Focused CLI/Page 04/Page 05 regression after the fix: **36 passed**; Ruff passed.

## Historical Branch B 5W1H amendment gate

The user requested a must-have Page 04/05 report and logs with 5W1H for each model, using existing upstream/Branch A-to-Branch B project outputs and the visual intent in `docs/spec-imporve-ui.md`, while preserving every current UI section. Research found a complete source-compatible primary audit and active supplemental evidence already exist, so the safe design reuses them rather than rerunning training. It will label this evidence historical and exposed; it will not misrepresent it as a strict unseen `training-validation/v1` pack or add Branch A cluster assignments as salary features.

The amended requirements, research, plan, data model, UI contract and tasks T039–T048 were explicitly approved by the user. Phase 9 implementation started with the following protected baseline: `core.py=d435a3dc…`, `training_validation.py=eef19641…`, `training_evidence_io.py=76d713ca…`, `pipeline.py=d4cb0692…`, `page06_prediction.py=867e8746…`, `config/project.yaml=da5d2325…`, `requirements.txt=8e2f2581…`, `uv.lock=96599229…`, `outputs/=20888ebc…`, and `artifacts/=bb8f9c2d…`.

## Phase 9 historical Branch B report implementation

Implemented without retraining or artifact regeneration:

- `load_compatible_training_audit()` now validates JSON-object events, projects at most 200 model-relevant raw events, retains total/relevant counts, and preserves byte-identical manifest/JSONL/export downloads.
- Page 04 builds exactly five deterministic candidate 5W1H records from the active supplemental candidate summary/folds plus compatible audit identity.
- Page 05 builds selected-family, Full 13 and fixed Top-2 5W1H records with inherited settings, historical locked-test metrics, q90 scope and non-promotion limitations.
- Both pages show a highlighted `Historical Branch B training report`, common-preparation/Branch A → Branch B flow, explicit no-cluster-feature statement, collapsed model records, bounded raw activity log and in-memory Markdown report download.
- Strict `training-validation/v1` remains independently unavailable with `RESERVE_KNOWN_EXPOSED`; historical values never populate strict fields.

Verification:

- RED loader tests first failed on absent event projection; RED 5W1H tests first failed on absent builders.
- Focused UV real-entrypoint check: **2 passed**, Streamlit health endpoint `ok`.
- Full pytest: **204 passed, 1 skipped**.
- Ruff and `git diff --check`: passed.
- Protected hashes exactly match T039 baseline, including `outputs/=20888ebc…` and `artifacts/=bb8f9c2d…`; no training/model output changed.
- Graphify refreshed: 1,577 nodes, 3,156 edges, 101 communities; the existing four zero-node JSON/config warnings remain non-blocking.

## Review status

- Correctness/security/performance review: no model deserialization, fit, predict, benchmark, publication, holdout-ledger access or write path is imported by the consumer. Discovery is fixed-path, symlink rejecting, checksum validated, strict-UTC, 100-run bounded, 50 MiB/file bounded and 100,000-row/table bounded. Invalid-candidate details are sanitized.
- Source independence verified through the complete four-state matrix on both pages: available/available renders both sources; unavailable/available preserves supplemental charts; available/unavailable preserves training evidence before the local supplemental error; unavailable/unavailable renders both local failure states without substitution.
- Browser review at 1280×800 and 1440×900: **pending** because Chrome DevTools/browser tooling is unavailable in this session; pixel acceptance is not inferred from AppTest.
- Two independent Vietnamese/English five-minute reader walkthroughs: **pending external reviewers**; automated tests are not substituted for this gate.
