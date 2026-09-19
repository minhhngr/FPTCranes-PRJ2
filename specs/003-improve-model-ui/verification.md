# Verification Record: Evidence-Backed Model UI

**Implementation approval**: Explicitly granted by the user on 2026-09-19 for T001–T067 and on the subsequent review turn for T068–T079, with the instruction to train only if additional evidence was actually needed.  
**Status**: T068–T078 and the automated/review portions of T079 are complete. Existing evidence was sufficient, so no pipeline retraining occurred. T044, T065 and T079's pixel/human portion remain pending because no interactive Chrome DevTools/browser automation or independent reviewers were available.

## Baseline preservation

Protected SHA-256 values captured before implementation:

```text
d435a3dcf846b5c5b778115e7ecdc7ca3e0741c27703d92b9a90bda28090f7c9  src/ai_job_market/core.py
d4cb0692ce3b028ff4d19b47aebe7c43c6f76bbc9529db6ce30d8e567ad867ee  pipeline.py
da5d23253b328f2594d345509c48931f5993a3d503588c10690e95f323f116a3  config/project.yaml
8e2f2581ea4d4ab13dc033c51c7f0654a9fee1e694c89762813464562934da4b  requirements.txt
96599229fba386c9268987794da072036c3f16443f7b828004fc56c8d9b00b27  uv.lock
7fa4b10fcecfbdd23823c0b0234786b67e0d7f28097c0e8d7faac41361c5cc45  data/raw/ai_jobs_market_2025_2026.csv
53b99e3d472a8d8627fc88eadc3b911d08304eafbf3e1d56704e404553f7830a  artifacts/model_bundle.joblib
b440df2b5b9ef3d71287035d5177fa200a659739dbf1f920581df1b6641dbebf  artifacts/metadata.json
23b31e5647b8191aee6d8acfe15032c9b7b20fb3ca7a6bffa56c01ad1b995a36  outputs/04_model_comparison/model_comparison.csv
631a76ec1b98741b64f00b031b09a6eb4672cb66d99656bc9373e101cf554786  outputs/05_best_model/locked_test_metrics.json
```

Pre-existing untracked files were preserved: `docs/spec-imporve-ui.md`, `specs/`, and `uv.lock`. The repository `.gitignore` already contains essential Python, environment, cache, artifact/output and universal editor patterns; no Docker/Node/Terraform/Helm project was detected, so no additional ignore file was created.

## Context review

- Existing Graphify graph was queried during specification/planning for `comparison prediction metadata bundle temporal tuning` and `temporal metadata bundle`; both traversals were truncated and followed by direct source inspection.
- Confirmed reusable public helpers in `src/ai_job_market/core.py`: model pipeline/preprocessor, temporal folds, candidate factory, metrics and encoded importance.
- Confirmed page routing contract `render(st, root, role)` in `streamlit.py` and inherited global upload training path in `src/components/data_source.py`; the latter remains outside this feature.
- Confirmed new generation logic must remain outside `core.py`; final hashes will prove preservation.

## Task progress and test evidence

### Red/green implementation

The feature was implemented in the approved phase order. Failing-first tests were added before the corresponding contract/policy/inference/producer behavior. Initial failures included missing `scenario_policy`, `ui_evidence_io`, `salary_inference`, producer and view modules; each slice was then brought green before the next UI slice. All generation tests used temporary workspaces. The real pack was not generated until producer, policy, inference and view tests passed.

New implementation is isolated in:

- `src/ai_job_market/ui_evidence_io.py`
- `src/ai_job_market/scenario_policy.py`
- `src/ai_job_market/ui_evidence_training.py`
- `src/ai_job_market/ui_evidence.py`
- `src/ai_job_market/salary_inference.py`
- `src/pages/model_evidence.py`

Pages 04–06 were reorganized behind the existing `render(st, root, role)` router contract. No page imports `run_pipeline`, invokes `.fit()`, or invokes the supplemental producer during interaction. The inherited administrator upload/full-pipeline control remains unchanged and outside feature scope.

### Real evidence generation and reuse

After fixture suites passed:

```text
PYTHONPATH=src .venv/bin/python -m ai_job_market.ui_evidence --workspace "$PWD"
→ generated ui-db11c8520f6498bf
PYTHONPATH=src .venv/bin/python -m ai_job_market.ui_evidence --workspace "$PWD" --check
→ valid: true
```

An earlier generated pack was correctly reported stale after a producer dependency was formatted; it was regenerated before being used. The final active pack contains 25 hashed files. A valid normal rerun was also exercised earlier and returned `status: reused`; CLI spies cover zero-fit unchanged reuse and pointer-preserving failure behavior.

Actual evidence highlights:

- five candidates over five paired temporal DEV folds;
- lowest candidate mean validation MAE: Random Forest, `$15,662.10`;
- full fixed configuration: DEV-CV MAE `$15,593.99`; historical-test MAE `$14,735.13`, R² `0.812690`, q90 `$40,067.75`, 298 rows;
- fixed Top-2: DEV-CV MAE `$13,484.86`; historical-test MAE `$13,829.48`, R² `0.820345`, q90 `$34,100.78`, 298 rows;
- three deterministic benchmark rows, all `historically_exposed=true` and `pristine=false`;
- inherited tuning snapshots were reused and labelled as non-nested DEV search ranked by R².

These are descriptive results from the real pack, not literals embedded in page source. Historical test metrics did not select or promote a candidate or Top-2 configuration.

### Automated verification

```text
UI_EVIDENCE_WORKSPACE="$PWD" pytest -q tests/test_ui_evidence_release.py
→ 1 passed

focused contract, release and UI suites
→ 43 passed

pytest -q
→ 95 passed, 1 skipped

UI_EVIDENCE_WORKSPACE="$PWD" pytest -q
→ 96 passed

ruff check <all changed Python source and tests>
→ All checks passed
```

The default-suite skip is the deliberately opt-in real-evidence test when `UI_EVIDENCE_WORKSPACE` is absent; the final opted-in full suite passed all 96 tests. `tests/test_model_ui_pages.py` opens all three pages through the real `streamlit.py` entrypoint and exercises Page 06 add/run/clear state invalidation.

A real Streamlit server was started successfully on port 8501 and `/_stcore/health` returned `ok`. AppTest timings were Page 04 `1.29s`, Page 05 `0.34s`, Page 06 `0.27s`; the Page 06 add/run/clear flow took `0.47s`. Separate model measurements were `0.450s` cold bundle load and `0.0062s` warm three-row inference. These are local measurements, not universal latency promises. Requested growth is intentionally a separate action and prediction call.

### Browser and reviewer acceptance

Interactive Chrome DevTools/browser automation was not available in this harness. Functional UI navigation and interaction were therefore checked with Streamlit AppTest plus a live-server health check, but pixel-level checks at 1280×800 and 1440×900, keyboard/screen-reader review, screenshots, download interaction in a real browser, and two independent human 60-second walkthroughs remain explicit release-acceptance items. No claim is made that these visual/human gates passed.

### Documentation and graph

`docs/MODEL_UI.md` records generation/reuse, source-to-view mappings, policy/exception rules, downloads and scientific limitations. README links to it. `graphify update .` completed with 1,136 nodes, 1,961 edges and 74 communities; it warned that three non-Python metadata/config files produced zero nodes, but the code graph update completed.

### Final preservation and review

Protected hashes were recomputed after implementation and exactly match the pre-implementation inventory above. In particular, `src/ai_job_market/core.py`, `pipeline.py`, configuration, source data, dependency files, baseline full bundle/metadata and primary comparison/test outputs were unchanged. The final diff has no new logic in `core.py` and no new dependency. Supplemental evidence is isolated below ignored `outputs/ui_evidence/` and `artifacts/ui_evidence/` paths with an atomic current pointer.

Five-axis review found no blocking correctness, architecture, security or performance issue:

- readers constrain paths to the workspace and verify manifest checksums before loading the trusted generated joblib;
- validation rejects unknown pairs, invalid/nonfinite years and mismatched model feature order before prediction;
- queue length is bounded and charts page visible rows;
- model/scientific qualifications are visible and downloads retain explicit provenance;
- no baseline fallback, arbitrary pickle path or silent partial batch result exists.

Final review verdict for the baseline: automated gates approve the implementation; visual/human acceptance remains pending as documented.

## Visual-priority amendment request

After reviewing the applied UI, the stakeholder reported that the visual result did not sufficiently match `docs/spec-imporve-ui.md`. On 2026-09-19 the living spec, plan, UI contract, checklist and tasks were amended to require:

- stable role-based semantic colors across Pages 04–06;
- historical-test dollar error as red columns and DEV-CV as the orange comparison line;
- blue/teal validation or prediction values, dashed red baseline references, amber uncertainty and red diamond actual markers;
- only P1 decision charts at full content width, with P2/P3 charts bounded, paired, compact, collapsed or requested;
- no restoration of unsupported stakeholder literals, pristine claims or causal fit diagnoses.

The amendment was initially specification-only. The stakeholder explicitly approved T048–T055 on 2026-09-19; implementation evidence follows.

### Visual amendment implementation

TDD red evidence:

```text
pytest -q tests/test_model_evidence_views.py
→ collection error: CHART_WIDTHS was not yet available from pages.model_evidence
```

The implementation then added a shared semantic palette and P1/P2/P3 container widths in `src/pages/model_evidence.py`, applied them to Pages 04–06, and added pure figure plus real-entrypoint AppTest assertions. Historical-test dollar errors now render as red columns; DEV-CV/training as orange lines; validation/prediction as blue/teal; uncertainty as amber whiskers; known actuals as red diamonds; baseline thresholds as dashed dark red. Supporting charts use 820-pixel P2 or 680-pixel P3 native containers, and tuning sensitivities use a two-column layout. No CSS or dependency was added.

Focused green evidence:

```text
pytest -q tests/test_model_evidence_views.py tests/test_model_ui_pages.py
→ 10 passed
```

Changing the shared figure builder legitimately changed the producer fingerprint and standalone HTML charts. `ui_evidence --check` reported the old pack stale, so the versioned supplemental pack was regenerated as required instead of being used stale:

```text
ui_evidence
→ generated ui-79dccd400d7b8c59
ui_evidence --check
→ valid: true
unchanged ui_evidence invocation
→ status: reused
```

Primary/baseline hashes remained identical. Final amendment checks:

```text
UI_EVIDENCE_WORKSPACE="$PWD" pytest -q
→ 100 passed
ruff check <changed visual source and tests>
→ All checks passed
Streamlit /_stcore/health
→ ok
graphify update .
→ 1,157 nodes, 2,000 edges, 74 communities
```

Five-axis review found no semantic-color amendment blocker: semantic roles are centralized and non-positional, non-color encodings remain, bounded containers preserve native responsive stacking, inference/evidence arithmetic is unchanged, and no new security or performance boundary was introduced. Interactive pixel/screenshot acceptance at both target resolutions remains pending because Chrome DevTools/browser automation is unavailable; AppTest verified all three real routed pages and the Page 06 prediction flow.

## Readability remediation request

The stakeholder subsequently reported that Pages 04–06 remain difficult to read: report elements overlap, important numbers are not visible, and a third party cannot understand the findings/conclusions. A specification-only audit inspected the source and real AppTest Plotly payloads.

Confirmed structural risks:

- `common.show_plot` overwrites every figure with the same 20/20/55/25 margins;
- Page 04 drift places 12 labels up to 19 characters in a 680-pixel view;
- Page 04 GB fold comparison emits 15 direct labels;
- each half-width Page 05 tuning chart emits 8–12 labels;
- Page 05 encoded importance places 15 feature labels against a 20-pixel left margin;
- pages expose caveats but not consistent derived `Finding / Why it matters / Limit / Decision-use` conclusions or a novice metric guide.

Detailed evidence is recorded in `ui-readability-audit.md`, research decisions R13–R16 and FR-029–FR-036. The stakeholder explicitly approved T056–T067.

### Readability implementation evidence

Failing-first checks established both defects:

```text
pytest tests/test_common_plot.py tests/test_model_evidence_views.py
→ missing conclusion/readability builders; shared renderer overwrote explicit geometry
```

Implemented behavior:

- `show_plot` now preserves explicit figure height, margins and hover mode, supplying defaults only when absent;
- figure builders define automargins, readable heights, ordinary numeric ticks and larger label margins;
- direct labels were reduced on compact comparison lines and Page 04 drift while exact values remain in hover/tables;
- Page 05 tuning charts stack at readable width instead of using half-width columns;
- Pages 04–06 now begin with a report question, evidence scope, derived takeaway and limitation plus a metric glossary;
- major sections include evidence-derived Finding / Why it matters / Limit / Decision-use conclusions;
- Page 06 conclusion is bound to the current result revision and disappears on queue mutation;
- visible tables use reader-facing names/units and pinned identities where supported; export contracts are unchanged.

Runtime Plotly inspection after remediation showed Page 04 candidate labels reduced 10→7 and drift labels 12→3. Page 04 figures now declare 430–470px heights and Page 05 figures 390–590px, with chart-specific margins up to 250px for long horizontal feature labels.

The shared generated-chart fingerprint changed, so the prior pack was correctly reported stale. One versioned pack was regenerated and then reused unchanged:

```text
ui_evidence
→ generated ui-b35971c9f8bcd166
ui_evidence --check
→ valid: true
unchanged ui_evidence invocation
→ status: reused
```

Final automated/runtime evidence:

```text
UI_EVIDENCE_WORKSPACE="$PWD" pytest -q
→ 112 passed
ruff check + ruff format --check (changed visual source/tests)
→ passed
Streamlit /_stcore/health
→ ok
graphify update .
→ 1,218 nodes, 2,136 edges, 75 communities
```

Protected baseline hashes remain identical to the pre-implementation inventory. Five-axis review found no blocker in automated scope: conclusions are deterministic and source-derived, geometry changes are presentation-only, queue invalidation prevents stale conclusions, no new dependency/CSS/security boundary was introduced, and inference/export calculations remain unchanged.

**Open gates**: T044 independent human walkthrough and T065 Chrome DevTools screenshots/DOM bounding boxes/console/accessibility review. Chrome DevTools MCP is unavailable in this harness; AppTest verifies behavior and Plotly structure but does not prove pixel-level non-overlap.

### Arrow serialization regression fix

A routed AppTest run exposed `pyarrow.lib.ArrowInvalid` on Page 01's Basic Clean comparison table. The `Raw` and `Basic clean` columns mixed integer values with strings such as `job_id present`, causing Arrow to infer `int64` before encountering text. A failing regression test was added in `tests/test_page01_data_basic_clean.py`; `build_clean_comparison_frame` now formats all comparison values as pandas string dtype before `st.dataframe`.

Verification:

```text
pytest tests/test_page01_data_basic_clean.py tests/test_model_ui_pages.py
→ 10 passed
runtime stderr scan
→ no ArrowInvalid or dataframe serialization fallback
UI_EVIDENCE_WORKSPACE="$PWD" pytest -q
→ 113 passed
ui_evidence --check
→ valid: ui-b35971c9f8bcd166
```

The fix changes presentation data typing only. It does not affect evidence/model artifacts, and no regeneration was required.

## Table and training-method amendment approval/preflight

The user explicitly approved T068–T079 and instructed that training should run only if more step/detail/metric/log evidence was needed. Preflight found that the active supplemental pack already contains current candidate fold metrics, fold membership and inherited tuning tables, while the complete primary-pipeline audit contains the manifest, JSONL trace and relevant fold/tuning CSV exports. Therefore the constitutional reuse rule applies: no training, tuning or evidence regeneration is needed.

Pre-edit checks:

```text
ui_evidence --check → valid: ui-b35971c9f8bcd166
current pointer SHA-256 → b0896bc0407ba9243282e112fe5c2dd182b372f9a251487aeab9d459e9bf2a04
current manifest SHA-256 → ed1ac8b844f1426858a5f4b38acb7b0218e585a1423b50a8b3f5ab623d0dedae
latest complete audit JSONL SHA-256 → 617e8d1afe4dced4a0b5d562ae4edd1e2ab8746413a06337d75081dffd15cb13
latest complete audit manifest SHA-256 → cc93b36641dc3774bbb12fc0e30d5994a411daaf77d9daf491f13589cecc47e7
```

All protected hashes still match the original baseline inventory. The amendment changes presentation and adds a validated read-only log consumer; it does not change training-producing code, source data, configuration, dependencies, evidence schema or the active model artifact contract.

### Table/training-method implementation evidence

Failing-first checks established the missing behavior:

```text
pytest tests/test_model_evidence_views.py
→ import failure: candidate_decision_table not implemented
pytest tests/test_model_evidence_views.py tests/test_training_audit_ui.py
→ import failures: temporal/tuning guide and compatible audit loader not implemented
pytest tests/test_model_ui_pages.py::test_page04_prioritizes_temporal_validation_guide_and_consistent_table_emphasis
→ priority temporal-validation guide absent
pytest tests/test_model_ui_pages.py::test_page05_explains_actual_tuning_and_distinguishes_applied_settings
→ priority tuning guide absent
```

Implemented behavior:

- new presentation-only `src/pages/model_training_presentation.py` owns compact table builders, actual-method summaries, byte-exact active-evidence downloads and the compatible historical audit reader;
- Page 04 displays a collapsed priority guide to the actual adjacent-block temporal splitter, fold-local preprocessing, MAE ranking, row-overlap result and non-random/non-expanding limitations;
- Page 05 displays a collapsed priority guide to four anchor trials plus 6/6/4/6 coordinate sweeps, five-fold CV R² ranking, saved initial-table selection provenance and non-nested/test-exposure limits;
- compact decision tables render Markdown emphasis with a visible legend; detailed dataframes retain plain reader-facing names, units and typed precision;
- active fold/tuning downloads are byte-for-byte source files from the validated current manifest;
- compatible primary-pipeline logs expose run IDs, manifest, relevant CSV exports and complete JSONL only after completeness, source fingerprint, safe-path and checksum validation;
- missing/incomplete/mismatched/symlinked audit inputs remain optional unavailable states and invoke no training.

An initial check correctly reported the pack stale while the new consumer helpers temporarily lived in the producer-fingerprinted `model_evidence.py`. The helpers were moved to the focused presentation-only module and `model_evidence.py` was restored byte-for-byte to SHA-256 `68f1ca554825c847a51766777b139536a9185e52bef8a2d59c4af01e422d20f6`. This preserved the producer contract and made regeneration unnecessary:

```text
ui_evidence --check
→ valid: ui-b35971c9f8bcd166
```

Final automated/runtime evidence:

```text
focused views/pages/audit/release suites → 46 passed
UI_EVIDENCE_WORKSPACE="$PWD" pytest -q → 123 passed
ruff check + ruff format --check → passed
git diff --check → passed
Streamlit /_stcore/health → ok
```

The active pointer/manifest and all protected baseline files retain their pre-amendment hashes. No pipeline run, model fit, tuning call or evidence regeneration occurred.

Five-axis review found no amendment blocker:

- **Correctness**: displayed method facts and emphasis derive from validated active tables/model parameters; current/audit downloads preserve exact bytes and provenance;
- **Readability**: compact Markdown tables and visible emphasis legends are separated from plain interactive detail; actual-method guides use explicit What/How/Why/Selection/Limits structure;
- **Architecture**: consumer-only helpers live in focused `model_training_presentation.py`, leaving the evidence-producing figure module byte-identical and avoiding needless artifact invalidation;
- **Security**: audit selection uses a fixed workspace root, source fingerprint, manifest completeness, contained nonsymlinked files, JSONL run identity and export checksums;
- **Performance**: audit files are bounded local evidence (current JSONL approximately 1.2 MB), no model work occurs, and routed AppTests remain within the existing test runtime.

`graphify update .` completed after final code verification with 1,272 nodes, 2,255 edges and 83 communities. It retained the existing warning that three metadata/config files produced zero nodes. Chrome DevTools screenshots, DOM/accessibility checks and independent human comprehension review remain unavailable, so T044/T065 and the corresponding portion of T079 are not marked complete.
