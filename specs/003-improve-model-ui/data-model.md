# Data Model: Supplemental Evidence and Controlled Scenarios

**Status**: Design only. No database or new service is introduced. JSON/CSV records and small typed Python structures implement these entities. Full wire contracts are in [evidence-pack.md](contracts/evidence-pack.md) and [scenario-ui.md](contracts/scenario-ui.md).

## Entity relationships

```text
Baseline source files + checkout/config/lock identities
  └── Evidence manifest (immutable; current pointer selects one)
       ├── Dataset snapshots → record IDs → fold membership
       ├── Frozen configurations → candidate/variant fold metrics
       │                          ├── historical test metrics/predictions
       │                          └── importance/uncertainty
       ├── Top-2 serving bundle + metadata
       ├── DEV scenario policy → observed pairs + title bounds
       └── Historical benchmark examples → real record IDs + variant predictions

Session context (workspace + evidence + policy)
  └── Validated queue (revision + unique scenario IDs)
       ├── Immutable result snapshot (same revision/model)
       │    ├── chart/table
       │    └── prediction-audit CSV
       ├── requested growth inputs/results (strict by default)
       └── original-schema scenario CSV
```

## 1. EvidenceManifest and ComponentStatus

- Identity: `schema_version`, `evidence_id`, creation time, optional existing source run ID.
- Provenance: named source hashes, dataset identities, source-code/config/lock/runtime identities, target, model/feature contracts, declarations and limitations.
- Component: name, dependency digest, `generated|reused|unavailable`, file references, availability reason and original producing evidence identity.
- Validation: all references stay within the workspace's allowlisted directories; hashes/schema/row counts are checked; no nonfinite JSON; current pointer references a completely validated manifest.
- A reused table retains its original producing `evidence_id`; the active manifest explicitly relates that component to the current pack and verifies identical dependency identity. It is not relabelled as newly generated. UI/session results use the active evidence ID plus model/configuration identity.
- No manifest assertion erases inherited historical test exposure. `legacy` source provenance and scientifically unverified historical claims remain distinct.

## 2. DatasetSnapshot and FoldMembership

- Snapshot: partition name (`development` or `historical_test`), exact file SHA-256, row count/order, temporal extent and feature schema.
- Record identity: partition + dataset hash + zero-based numeric source-row offset; unique even with duplicate row contents. It is not an original job ID.
- Membership: fold ID, `train|validation`, ordered record IDs, row counts and actual period labels.
- Validation: training/validation indices nonempty and disjoint within a fold; all fold IDs refer only to DEV; no historical test row enters preprocessing/model fit. Adjacent row blocks may share month labels, which is disclosed instead of claiming strict month separation.
- Folds are materialized once and shared by five candidates and both fixed final variants. The evaluation declaration records their identity before new test-target scoring.

## 3. ModelIdentity and EvaluationResult

- Model: model ID, family, feature variant, ordered input names, class/actual parameters, seed, configuration digest, applied/historical selection relationship and optional bundle reference/hash.
- Candidate defaults, selected full settings and Top-2 fixed settings are distinct identities, even if the family name matches.
- Evaluation: dataset/fold identity; train/validation/test partition; MAE, RMSE, R², MedAE; rows, dimensions and measured durations; missing-metric reason where appropriate.
- Aggregation: unweighted fold metric means; MAE population SD. No mixing final fitted-train error with fold-train error or baseline CV with tuned CV.
- Test metrics are historical diagnostics with `selection_use=none`; no result writes back a new selected full model or changes Top-2 parameters.

## 4. ImportanceEvidence and EmpiricalBand

- Importance: model/configuration, scope (`encoded|raw_family|raw_permutation`), feature identity, fold or test partition, value and variation; include repeats/seed for permutation.
- Raw-family drift derives from per-fold family sums before aggregation. Encoded and raw-family weights are never silently conflated. Negative permutation means are allowed.
- Band: model identity, test snapshot, q=0.9, method=linear, absolute-error quantile in USD, coverage denominator and fraction, same-population diagnostic limitation.
- Band is neither a formal confidence interval nor validated extrapolation coverage. UI cannot transfer full q90 to Top-2 or vice versa.

## 5. ScenarioPolicy

- Identity: schema version, content hash, DEV/metadata source identity, default mode=strict.
- Pair entries: category, title, DEV support count; allowed enums exactly match compatible metadata.
- Title entries: finite minimum, maximum, median and DEV support count, aggregated without salary/test data. Median may be fractional.
- Global policy: manual exceptions forbidden; benchmark/curve experience-only acknowledgement permitted within 0–15; max queue 100.
- Validation: min ≤ median ≤ max, valid supported title, observed pair, finite non-boolean years. Out-of-range experience has an explicit error, not a clamped silent replacement.

## 6. BenchmarkExample

- Real evidence record ID, source offset, immutable role/category/experience and verified original attributes.
- Optional actual salary (known for generated historical examples), full/Top-2 predictions/errors, model IDs.
- Selection: source-order first up to three strict-eligible historical rows, no target/error/accuracy filtering.
- Provenance flags: `included_in_benchmark=true`, `historically_exposed=true`, `pristine=false`.
- Fewer than three eligible examples is an availability fact, not permission to fabricate missing rows. These examples remain in full historical test metrics.

## 7. Scenario and ExceptionAcknowledgement

- Scenario ID unique within session; source=`manual|benchmark`; optional immutable benchmark record ID; title, category, experience.
- Audit-only actual and known original attributes are populated from benchmark lookup, not trusted from browser payload.
- Validation mode=`strict|experience_exception`; exception reason fixed by policy, acknowledgement bound to context, model evidence and benchmark/curve request.
- Manual rows cannot obtain exception mode by changing a flag in session state. Benchmark values must match their recorded profile; categorical checks are never excepted.
- Duplicate profiles are allowed with separate scenario IDs, preserving table/chart/export alignment.

## 8. SessionContext, Queue and PredictionBatch

- Context: resolved workspace, active evidence ID, policy hash and serving model ID.
- Queue: ordered scenarios, revision and next ordinal. Cap 100. No mutable global state.
- Batch: context, immutable queue revision/hash, result rows and generation time. Result rows contain raw prediction, clipped lower bound, upper bound, q90/basis, optional actual/absolute error/signed variance, validation mode/reason.
- Invalid/nonfinite output or wrong shape fails the batch; no partial new results replace a valid snapshot.
- Export rounding is presentation-only. Unknown actual → blank errors; zero actual → undefined signed percentage; negative/nonfinite actual → validation error.

### State transitions

| Event | Prior state | Next state / invariant |
| --- | --- | --- |
| Load valid context | New session | Empty queue, no results, strict mode |
| Builder selection changes | Any | Reset dependent controls as needed; queued immutable rows unchanged |
| Add / quick-load valid record | Queue size <100 | Append unique ID; revision++; clear previous result/curve |
| Add invalid record | Any | Visible row-specific error; no queue/result mutation or prediction |
| Run valid nonempty queue | Ready | Validate all rows, one batch predict, store matching snapshot |
| Run invalid queue/model | Any | No prediction; show actionable reason; no partial result |
| Request growth | Current successful snapshot | Validate derived grid, predict requested points, label extensions |
| Disable exceptions | Queue/curve may contain exceptions | Invalidate results; visibly remove/reject affected queue rows and recompute only after explicit Run |
| Workspace/evidence/policy/model changes | Any | Clear queue, results, curves and acknowledgements; explain context reset |
| Clear | Any | Empty queue and no results/curves/acknowledgements |
| Unrelated rerun | Valid matching snapshot | Retain queue/results; no fit or implicit predict |

## 9. ReadabilityProfile and EvidenceConclusion

- **ReadabilityProfile**: figure/view ID, P1/P2/P3 priority, minimum width/height, margins, axis automargin/tick format, direct-label policy, dense fallback, orientation and visible Top N where applicable.
- Profile validation: required labels must have a non-hover representation; dense scatter text is disabled; compact views cannot combine unrestricted text labels with long categories; shared rendering preserves explicit geometry.
- **EvidenceConclusion**: conclusion ID, active evidence/model/population identity, `finding`, `why_it_matters`, `limit`, optional `decision_or_use`, and named source fields.
- Conclusion validation: all scientific values derive from finite compatible evidence; direction and units are explicit; missing requirements yield a named unavailable state; queue/result conclusions bind to the current context/revision and invalidate with results.
- Educational glossary entries are static definitions, not evidence conclusions. They contain no current winner, metric value or model verdict.

## 10. Exports and chart views

- Prediction audit v1: explicit schema/model/evidence/policy identity, scenario and benchmark IDs, input profile, strict/exception state, money/bounds/errors with declared precision.
- Original source-schema export: exactly 25 original columns with category mapped to `AI Engineering`; unknown attributes remain blank and synthetic scenario IDs clearly identified.
- Charts: derived, not independent scientific entities; preserve model/partition/configuration and row identities. Scenario chart paging affects display only, never table/export contents. Figure geometry and label selection change presentation only; they do not alter authoritative metric rows.
- Cached evidence and models are immutable shared resources keyed by workspace/hash; per-user queues, acknowledgements and results are never shared cache entries.
