# Optimization Notes

The project prioritizes reproducibility, auditability and safe reuse on future schema-compatible data.

1. **Leakage control** — target-adjacent salary fields are blocked; locked test is isolated before fitting; preprocessing is TRAIN/DEV-only.
2. **Output-first UI** — every analytical page reads pipeline artifacts. Filters slice evidence but never fit/tune models.
3. **Interactive evidence** — Plotly provides hover values, zoom, legend focus, dynamic metric selection and category filters.
4. **Feature traceability** — encoded columns are linked back to original fields; scaling before/after evidence is stored; skills are tokenized using the training vocabulary.
5. **Unsupervised balance** — feature families are independently encoded and scaled by sqrt(dimension) before PCA and KMeans/GMM search.
6. **Temporal model selection** — candidate families share identical expanding monthly folds, including a Dummy Median reference floor and runtime measurement.
7. **Locked-test diagnostics** — actual-vs-predicted, residual tail, raw permutation importance, encoded importance and subgroup error review are separate evidence layers.
8. **Serving consistency** — preprocessing + estimator are serialized in one bundle and metadata controls categories, ranges, skill vocabulary, field order and empirical error band.
9. **New-data adaptation** — when the preferred holdout is no longer latest, the latest available year-month is used; unseen categorical values are handled safely by fitted encoders.
