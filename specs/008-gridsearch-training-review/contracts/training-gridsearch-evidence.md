# Contract: training GridSearchCV evidence

## Producer contract

The revised training evidence schema must add `training_method_version=gridsearchcv-temporal/v1` and a complete GridSearchCV result projection. Each result must identify its temporal context, scoring direction, inner folds, parameter values, rank, fold-level and mean/std score, timing, status/failure reason, and source reference. `n_estimators` and `max_depth` are mandatory declared dimensions. Older manual-search manifests are invalid for revised-method claims.

The pack must include a detailed English evidence-derived `training.log` whose numbers come from the authoritative run tables. It must answer 5W1H and record the ordered split, fold, five-model, fit/runtime, selection, tuning, final evaluation, explainability, uncertainty, and status steps. `events.jsonl` remains the separate raw structured lifecycle record.

The producer must reject a run if a temporal search context includes a row outside its parent TRAIN population, if chronology/overlap validation fails, or if an evaluation reserve is known exposed. Failed/ineligible runs produce a safe status and do not publish partial final packs.

## Consumer contract

Pages 04–05 accept only a compatible final pack. They must display the method/schema status and must treat older manual-search packs as `invalidated` for revised-method evidence. They do not deserialize model bundles or execute training operations.

## Serving compatibility

Page 06 continues accepting only its established validated Top-2 serving metadata/bundle. A revised training-validation pack does not replace its serving source. If future changes alter the serving metadata schema, Page 06 must require an explicit compatible version and reject old metadata/bundles.
