# Execution-to-Evidence Coverage Inventory

**Source baseline**: [research.md](research.md), 2026-09-18 working tree. Line references are navigation aids; function names and table names are the durable anchors. **Planned**, not implemented.

## Coverage rule

Each executed stage has a start and exactly one completed/failed/cancelled end. A deliberately unexecuted branch has `skipped` with a reason. A process killed without cleanup leaves open operations and must not be retroactively marked complete. File creation is not proof of stage completion. All source operations remain in their existing order. Additional export failures affect evidence completeness, not scientific results.

Level 1: stage/model action. Level 2: counts, candidate outcomes, stage summaries, tuning progress. Level 3: fold/trial/segmentation candidate detail. Level 4: exact parameter/feature/membership/derivation evidence and export references. Warnings/errors bypass depth filtering.

## Inventory

| ID / actual location | Current observation gap | Planned evidence / completion boundary | Verification task |
|---|---|---|---|
| PREFLIGHT — `pipeline.py:_preflight`, `main` | Runs before audit session; only text | Preserve existing validation and exit 2; record safe preflight outcome in current session once created, or clear terminal `not started` on validation failure. No training stage claimed. | T011 |
| RUN — `core.py:run_pipeline`, `_run_pipeline_impl` | Config/hash coverage partial; audit initialization can abort training | Source/config fingerprints, actual seed/versions/revision+dirty state, output locations, separate training/evidence states; file initialization/close warnings with continuation only for logging I/O errors | T004–T006 |
| C1 — `_run_pipeline_impl:3321`, `read_raw`, `profile_dataframe` | Artifact events only | Raw source/canonicalization description, row/column counts, schema/profile evidence; end after raw profile save | T013 |
| C2 — `basic_clean:432` and caller | No count arithmetic in console | Raw minus invalid-category minus duplicates equals clean rows; identifier column removal; no raw invalid job IDs copied into new logs; end after clean data/audit writes | T013 |
| C3 — `contradiction_outputs`, `stage1_detailed_outputs`, target summary | Many output paths without action/count context | Quality-check purpose, counts and returned report references; existing whole-clean-data descriptive scope labeled, not called unseen validation | T013 |
| C4 — `feature_policy_table:804` | No explanation of admitted/blocked features | Target, 13 current primary inputs, blocked reasons, actual feature table; end after policy save | T013 |
| C5 — prepared base and `derive_skill_count` | Engineered-feature derivation absent | Prepared row count, new skill_count = unique normalized skill token count; before/after columns; end after prepared save | T013 |
| B1–B3 / split — `temporal_split:849` and caller | No first-class partition event | Configured and actual test period/rule, DEV/test counts/percent denominator, safe DEV fingerprint; do not resplit | T018 |
| B1–B3 / ablations — `run_ablation:1190` | Fold logs exist; candidate summary absent | Each actual variant's features, parent identity, fold events, summary/variability and full variant table; maintain four variants and existing order | T019, T022 |
| B1–B3 / correlations — `encoded_correlations_train:1383` | Combined fit_transform has no audit | DEV-only fit_transform start/end, input/encoded counts, correlation method and table link; no extra transform | T019 |
| B1–B3 / readiness — `make_salary_preprocessor(...).fit`, DEV/test `.transform`, `stage2_detailed_outputs` | Unobserved fit/transforms | Separate fit and transform-only messages; shapes, vocabulary/names from fitted preprocessor, readiness counts; stage completes after `08_training_readiness.json` | T019 |
| A1–A8 / encoder — `run_segmentation:1922` | Long stage mostly opaque | DEV-only family-balanced encoder action, input/encoded dimensions and no salary target as input | T014 |
| A1–A8 / representations — `build_representation_specs` call, representation loop | No representation progress | O1 build start/end; each actual R0–R4 evaluation start/end with raw/latent feature counts, returned candidate/stability metrics and reference tables | T014 |
| A1–A8 / O1 selection + official search | O1 screening and official rerun can be confused | Distinct IDs for initial evaluation and official O1 evaluation; actual selection evidence and resulting K/algorithm | T014 |
| A1–A8 / O2 — `_build_correlation_selected_space`, `evaluate_candidates` call | Selection dimensions unclear | DEV-only correlation filter threshold, before/selected dimensions, transform scope, O2 candidate metrics and resampling export | T014 |
| A1–A8 / option decision, evidence, bundle | Marker currently precedes bundle write | Actual O1/O2 decision, silhouettes/stability/balance, assignments count, evidence tables; end only after cluster bundle write | T014, T027 |
| B4 / ladder — `candidate_models:1176`, comparison loop | No coherent model start/end or aggregate event | Five actual candidates, feature set/params, fold summaries, model timing, mean/population SD from existing scores; rank by actual MAE | T022–T024 |
| B4 / fold operations — `evaluate_model_cv:1054` | No unique evaluation ID; failed operation not closed | Split definition/reuse; fold fit/predict/score start/end/failure, input counts and fitted encoded-feature count; preserve one fit+predict per fold | T018–T019 |
| B4 / family ablation — `run_feature_family_ablation:1216` | Feature policy exceptions hidden by names | Actual four feature sets (including experimental experience_level), comparison to A0 with formula, clearly experimental not deployed | T022–T024 |
| B4 / importance — `random_forest_importance_by_fold:1264` | Features field overloaded as integer in extraction event | Explicit encoded_feature_count; fit/extract boundaries and drift table; `metrics_status=not_computed`, no added validation scoring | T019 |
| B5–B6 / tuning — `tune_random_forest_manual_steps:2761` | Trial parent/seed/progress incomplete | Five groups, 26 actual RF trials; parameter source, constant settings, start/end/progress, fold refs and winner per group; stop on existing exception, no continue-on-error invention | T023–T024 |
| B5–B6 / selection — comparison first row, `final_model_from_selection:3005` | Non-RF tuning skip absent | MAE family choice; RF R² tuning; s1 as final source; actual applied parameters and difference versus later-stage reported settings; explicit non-RF skip | T023–T024 |
| B5–B6 / final fit/predict/score — `finalize_salary_model:3061` | Some actions logged, no final model context | DEV-only final fit, selected-model-only test prediction and four metrics; other candidates `not_evaluated` | T024 |
| B5–B6 / diagnostics — q90, permutation, encoded importance, subgroup/error slices | q90 and report stages not explicit | Existing empirical q90 error-band calculation (not calibrated coverage), permutation repeat count, feature reliance caveat, subgroup table refs; end after locked-test metrics save | T024 |
| B7 / serialization — `_run_pipeline_impl` joblib dump/load sites | CSV/JSON wrappers do not cover joblib | Start/end/error for each actual write and reload, artifact path/role, no extra dump/load, existing alias names unchanged | T027 |
| B7 / reload equivalence + examples | Existing two predicts unobserved | Log both current `.head(20)` predictions and measured max difference/tolerance; if passed false, show failed check, not blanket PASS; preserve existing scientific flow | T027 |
| B7 / prediction summary | Could print raw predictions | Link existing examples; show available aggregate values/empirical error band; no raw row duplication in new logs; end after prediction summary save | T027 |
| I1 — integrated grouping, `predict_segments` call | Only table writes | Input/assignment/group counts, existing segment prediction start/end, group aggregation definitions and outputs; no refitting | T027 |
| F — status and summary writes | Legacy status table hardcodes PASS | Observe actual completion, source fingerprint and result pointers; existing table remains unchanged but must not be treated as audit truth; current-run audit manifest is authoritative for observation | T027 |
| IO — `save_csv:180`, `save_json:141` | Repeated basename step IDs; no failure event | Full workspace-relative path + unique operation ID, start/completed/failed; report sizes/counts, not raw payloads; evidence exporter uses direct writes to avoid recursive audit | T006, T028–T029 |
| CLI completion — `TerminalProgress`, timing save, final summary | `finalize_unobserved` fabricates PASS | Event-based status/timing; no silent completion of missing events; preserve heartbeat/no-heartbeat and exit semantics; logging-only timing failures warn | T011–T012, T031 |

## Deliberate boundaries

- `segmentation_robustness.py` and `segmentation_stability.py` are delegated algorithms. The caller-level boundaries and their returned tables are observed; their individual resample/solver iterations are not separately instrumented. Record this limitation rather than claim per-fit tracing inside Branch A.
- No per-tree, per-boosting-iteration, optimizer or third-party-library debug flood. Existing third-party warnings remain visible.
- No UI edits, extra model evaluation, plot regeneration or new research experiment. Existing prediction/raw-row CSV artifacts remain as they are; the feature adds safe summary/membership exports rather than duplicating the dataset.
- Every new event type must be classified in tests by depth and required fields. New training operations discovered during implementation require this inventory and relevant tasks to be updated before implementation continues.
