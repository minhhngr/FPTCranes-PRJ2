# Quickstart and Verification: Model UI Improvement

**Status**: Baseline/color implementation has completed. Commands also define the proposed readability-remediation verification; do not implement T056–T067 until separately approved.

## 1. Preconditions and preservation

From the repository root:

```bash
git status --short
git branch --show-current
.venv/bin/python --version
```

Expected feature branch: `005-improve-model-ui`; feature directory is independently `specs/003-improve-model-ui/`. Preserve existing untracked `docs/spec-imporve-ui.md`, prior `specs/` and `uv.lock`; do not `git add .`, replace locks or reset unrelated work. Auto-commit hooks are disabled by local configuration.

Before implementation, create `specs/003-improve-model-ui/verification.md` with explicit task approval and a SHA-256 inventory of `src/ai_job_market/core.py`, `pipeline.py`, source data, config/lock, baseline output packs and existing artifacts. Store machine-readable hashes outside those protected packs. Capture actual working-tree contents, not only HEAD. Record existing provenance limitations.

Re-query the existing Graphify graph; follow direct source inspection when results are truncated. After code changes, refresh the graph as documented in Section 7.

## 2. Failing-first focused tests

Tests listed here will be created by the tasks; they do not exist yet. Write the relevant test first, record the expected failure, then implement its smallest matching behavior.

```bash
.venv/bin/python -m pytest -q tests/test_ui_evidence_contract.py tests/test_scenario_policy.py
.venv/bin/python -m pytest -q tests/test_ui_evidence_training.py tests/test_ui_evidence_cli.py
.venv/bin/python -m pytest -q tests/test_salary_inference.py tests/test_model_evidence_views.py
.venv/bin/python -m pytest -q tests/test_model_ui_pages.py
```

Producer tests use tiny deterministic data, fake/counting estimators where possible and temporary workspaces. Controlled fitting in producer tests is permitted after implementation approval; no test may overwrite baseline outputs. UI/policy/serving tests monkeypatch fit/tune/producer entrypoints to fail if called and assert invalid input makes zero predict calls.

Required cases:

- Matching/missing/corrupt/escaping artifacts; mixed model/partition IDs; partial publication failure; unsupported schema and local-pickle path; stale dependency fingerprints.
- Train/validation scores from one fit; all five models share membership; test-target mutation cannot change configurations, policy or selected benchmark source-row offsets (snapshot hashes/record IDs may change); no test rows fit preprocessing/model.
- Full/Top-2 feature orders and params; recorded/reloaded Top-2 predictions; full source bundle compatibility; unchanged repeat has zero fit calls; presentation/policy rebuild never refits unaffected models.
- Observed pair membership, fractional median 7.5, single-value bounds, unknown title, nonfinite/bool years, stale state, 0/100/101 rows, immutable benchmark inputs and actuals.
- Strict default and opt-in experience-only exception paths; unknown pairs remain invalid; acknowledgement expires when policy/evidence changes.
- Point/bounds math, clipped lower bound, negative/invalid output ordering, actual outside band, unknown/zero/invalid actual, signed versus absolute errors, exact CSV columns and correct original `AI Engineering` mapping.
- Real entrypoint login/role/workspace routing, default tabs/sections, dependent widget updates, add/run/clear, queue mutation and context resets, optional missing tuning history and unavailable Top-2.
- Per-figure margin/height/automargin and label-density profiles; shared rendering preserves explicit geometry; long labels and negative R² remain visible.
- Evidence-derived conclusions under changed rank/delta/population fixtures, named unavailable conclusions, and Page 06 conclusion invalidation after queue/context mutation.
- Reader-facing table names/units and exact underlying download preservation.

AppTest should use `streamlit.py` with controlled test-only login/workspace fixtures for integration. Direct renderer fixtures may supplement it for isolation; do not mistake importing a page module for exercising the entrypoint. No production authentication bypass is added.

## 3. Build/check the real supplemental pack once

After tests for the new producer pass and implementation is approved:

```bash
# Inspect whether compatible evidence already exists; nonzero when absent/invalid.
PYTHONPATH=src .venv/bin/python -m ai_job_market.ui_evidence --workspace "$PWD" --check

# Generate only missing/invalidated components in the NEW namespace.
PYTHONPATH=src .venv/bin/python -m ai_job_market.ui_evidence --workspace "$PWD"

# Read-only structural/source-integrity validation; no fitting or predicting.
PYTHONPATH=src .venv/bin/python -m ai_job_market.ui_evidence --workspace "$PWD" --check
```

This new-artifact generation is necessary because the feature adds a producer/consumer contract and a two-feature model. It is not permission to run the complete main pipeline, change legacy artifacts, or retrain unchanged valid components. No `python pipeline.py` command is needed for this feature's normal verification.

The command must show actual evidence ID, component reused/generated status and any inherited/unavailable tuning evidence. Do not promise that regenerated metrics equal document examples. Confirm existing raw/prepared split, protected packs and `core.py` retain their baseline hashes.

Prove zero-fit reuse in the isolated CLI test suite with fit counters. Also run a second unchanged real normal invocation; its summary must report reuse, and the real-pack check below must verify equivalent saved identities/predictions without fitting. Do not regenerate artifacts a third time for screenshots.

## 4. Saved-artifact consumer and regression checks

`tests/test_ui_evidence_release.py` must be read/predict-only and skip with an explicit reason unless a workspace is deliberately supplied; it never generates its own real artifacts:

```bash
UI_EVIDENCE_WORKSPACE="$PWD" .venv/bin/python -m pytest -q tests/test_ui_evidence_release.py
.venv/bin/python -m pytest -q tests/test_release_contract.py
.venv/bin/python -m pytest -q
```

Record skips, failures and test counts exactly. Existing unrelated producer-heavy tests must use their existing isolated workspaces/fixtures. If a full-suite failure is inherited, document it; do not silently weaken unrelated scientific tests. Only outdated UI widget-count/required-prose assertions may be replaced with equivalent behavior/contract assertions for Pages 04–06.

Targeted lint/format verification (use the repository's installed tool, no package installation):

```bash
.venv/bin/python -m ruff check src/ai_job_market/ui_evidence*.py src/ai_job_market/scenario_policy.py src/ai_job_market/salary_inference.py src/pages/model_evidence.py src/pages/page04_model_comparison.py src/pages/page05_best_model.py src/pages/page06_prediction.py src/pages/common.py tests/test_ui_evidence*.py tests/test_scenario_policy.py tests/test_salary_inference.py tests/test_model_evidence_views.py tests/test_model_ui_pages.py
.venv/bin/python -m ruff format --check src/ai_job_market/ui_evidence*.py src/ai_job_market/scenario_policy.py src/ai_job_market/salary_inference.py src/pages/model_evidence.py src/pages/page04_model_comparison.py src/pages/page05_best_model.py src/pages/page06_prediction.py tests/test_ui_evidence*.py tests/test_scenario_policy.py tests/test_salary_inference.py tests/test_model_evidence_views.py tests/test_model_ui_pages.py
git diff --check
```

Do not reformat unrelated files just to improve a broad lint report.

## 5. Application and browser walkthrough

Check whether the app is already running; ask before launching a new server. With approval:

```bash
PYTHONPATH=src .venv/bin/streamlit run streamlit.py --server.port 8501
```

Use the existing login flow; do not modify auth or invoke the global upload/process button. That inherited button can run training and is explicitly outside the new workflow.

At **1280×800** and **1440×900**, record screenshots/evidence for:

1. Page 04: five chips, four tabs, chart-before-table order, labels, negative R² axis, Dummy reference, actual fold counts and model params. Find winning CV family/baseline within 60 seconds; distinguish saved selection if different.
2. Page 05: six chips/five sections, separate R² scale, final-config CV mapping, 298-row historical scope when using current data, collapsed methodology, four sweep charts, raw/encoded distinction, Top-2/full comparison and historical—not pristine—example labels. Locate MAE/count/band caveat within 60 seconds.
3. Page 06: dependent category/title/years updates, fractional default, no free text, strict badge, eligible quick-load actual preservation, immutable benchmarks and disabled unavailable-model Run.
4. Queue sizes 1, 3, 100: unique labels, ten-row chart pages, full table/export, actual markers outside bands where evidence dictates, signed variance and CSV column/precision reconciliation.
5. Default bounded growth curves; explicit extension acknowledgement; dashed/labelled exception points and historical-band warning; turning exception mode off invalidates affected results.
6. Add after Run, Clear, change workspace/evidence and new browser session: no stale results, acknowledgements or cross-session rows.
7. Keyboard focus, accessible labels, legends/shape distinctions and direct chart labels without clipping/overlap. Use DOM bounding boxes to verify zero required-label intersections/viewport overflow; capture the console and accessibility tree. Confirm downloaded files contain the displayed snapshot and correct `AI Engineering` source category.
8. Third-party comprehension: without downloads/source, identify each page's question, main finding, two supporting values and key limitation within 90 seconds. Verify every section conclusion follows Finding / Why it matters / Limit / Decision-use and matches the visible evidence.
9. Missing/corrupt fixture workspace: actionable local unavailable states, no crash, no silent full-model fallback or fitting.

Use configured browser tooling or a human review. If unavailable, leave browser acceptance explicitly pending. AppTest is not screenshot, DOM or download verification.

## 6. Performance and evidence reconciliation

Measure with a documented machine/browser and fixed three-scenario snapshot:

- Cold artifact verification/model load.
- Warm batch prediction wall time.
- Warm end-to-end visible result time (target <1 s).
- Optional growth computation separately; hidden/unrequested curves must not run.
- 100-scenario view rendering/pagination and audit-export completeness.

Reconcile every scientific number to manifest/table/derived formula at display precision; assert full/Top-2 test row IDs are identical and examples remain included in that population. Label the model's empirical q90 basis and unsupported extrapolation coverage. No performance/scientific claim passes merely because a caption prints it.

## 7. Documentation, graph and completion gate

Update `docs/MODEL_UI.md`, README and feature `verification.md` with actual commands/results, source-to-chart references, evidence IDs, caveats and any remaining manual gates. Keep the spec/plan/contracts/tasks synchronized if implementation reveals a material change; ask before changing scope.

```bash
graphify update .
git diff --check
git status --short
```

If the installed graph CLI differs, use its documented incremental-update equivalent; record the exact command or a valid scan/query-only exception. Do not refresh training merely to refresh Graphify.

Mark tasks complete only when their verification evidence exists. Before merge: inspect the actual diff, confirm no generation logic was added to `core.py`, all baseline hashes remain unchanged, new artifact fingerprints match current producing code/contracts, tests are recorded accurately and the human has reviewed outstanding scientific/UI limitations.
