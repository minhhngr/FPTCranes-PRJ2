from __future__ import annotations

import random
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.inspection import permutation_importance

from .core import TARGET, extract_encoded_importance, make_model_pipeline, regression_metrics


def _raw_family(encoded: str, features: list[str]) -> str:
    suffix = encoded.split("__", 1)[-1]
    for feature in sorted(features, key=len, reverse=True):
        if suffix == feature or suffix.startswith(f"{feature}_"):
            return feature
    if encoded.startswith("skills__"):
        return "required_skills"
    return "unknown"


def evaluate_frozen_models(
    dev: pd.DataFrame,
    features: list[str],
    models: dict[str, BaseEstimator],
    splits: list[tuple[np.ndarray, np.ndarray, str]],
    *,
    capture_importance: bool = True,
    feature_variant: str = "candidate",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit each frozen model once per fold and score its train/validation rows."""
    fold_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    x = dev[features]
    y = dev[TARGET]
    for model_name, base in models.items():
        params = clone(base).get_params(deep=False)
        configuration_id = _configuration_id(model_name, features, params)
        for fold_id, (train_idx, validation_idx, period) in enumerate(splits, start=1):
            pipeline = make_model_pipeline(clone(base), features)
            started = time.perf_counter()
            pipeline.fit(x.iloc[train_idx], y.iloc[train_idx])
            fit_time = time.perf_counter() - started
            started = time.perf_counter()
            train_prediction = np.asarray(pipeline.predict(x.iloc[train_idx]), dtype=float)
            train_predict_time = time.perf_counter() - started
            started = time.perf_counter()
            validation_prediction = np.asarray(
                pipeline.predict(x.iloc[validation_idx]), dtype=float
            )
            validation_predict_time = time.perf_counter() - started
            train_metrics = regression_metrics(y.iloc[train_idx], train_prediction)
            validation_metrics = regression_metrics(y.iloc[validation_idx], validation_prediction)
            encoded_count = len(pipeline.named_steps["preprocess"].get_feature_names_out())
            row: dict[str, Any] = {
                "model": model_name,
                "model_id": f"{feature_variant}:{model_name}",
                "configuration_id": configuration_id,
                "feature_variant": feature_variant,
                "fold_id": fold_id,
                "validation_period": period,
                "train_rows": len(train_idx),
                "validation_rows": len(validation_idx),
                "encoded_feature_count": encoded_count,
                "fit_time_s": fit_time,
                "train_predict_time_s": train_predict_time,
                "validation_predict_time_s": validation_predict_time,
            }
            row.update({f"train_{key}": value for key, value in train_metrics.items()})
            row.update({f"validation_{key}": value for key, value in validation_metrics.items()})
            fold_rows.append(row)
            estimator = pipeline.named_steps["model"]
            if capture_importance and hasattr(estimator, "feature_importances_"):
                for encoded, value in zip(
                    pipeline.named_steps["preprocess"].get_feature_names_out(),
                    estimator.feature_importances_,
                ):
                    importance_rows.append(
                        {
                            "model": model_name,
                            "model_id": row["model_id"],
                            "configuration_id": configuration_id,
                            "feature_variant": feature_variant,
                            "fold_id": fold_id,
                            "validation_period": period,
                            "encoded_feature": str(encoded),
                            "raw_family": _raw_family(str(encoded), features),
                            "importance": float(value),
                        }
                    )
    folds = pd.DataFrame(fold_rows)
    summaries: list[dict[str, Any]] = []
    for (model_name, model_id, configuration_id), group in folds.groupby(
        ["model", "model_id", "configuration_id"], sort=False
    ):
        result: dict[str, Any] = {
            "model": model_name,
            "model_id": model_id,
            "configuration_id": configuration_id,
            "feature_variant": feature_variant,
            "effective_folds": int(len(group)),
            "validation_MAE_SD": float(group["validation_MAE"].std(ddof=0)),
            "fit_time_mean_s": float(group["fit_time_s"].mean()),
            "predict_time_mean_s": float(group["validation_predict_time_s"].mean()),
        }
        for prefix in ("train", "validation"):
            for metric in ("MAE", "RMSE", "R2", "MedAE"):
                result[f"{prefix}_{metric}_mean"] = float(group[f"{prefix}_{metric}"].mean())
        summaries.append(result)
    return folds, pd.DataFrame(summaries), pd.DataFrame(importance_rows)


def _configuration_id(name: str, features: list[str], params: dict[str, Any]) -> str:
    import hashlib
    import json

    serializable = {
        key: value
        for key, value in params.items()
        if isinstance(value, (str, int, float, bool, type(None)))
    }
    raw = json.dumps(
        {"name": name, "features": features, "params": serializable},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def evaluate_test_models(
    dev: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    models: dict[str, BaseEstimator],
    *,
    feature_variant: str = "candidate",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    fitted: dict[str, Any] = {}
    for name, base in models.items():
        pipeline = make_model_pipeline(clone(base), features)
        pipeline.fit(dev[features], dev[TARGET])
        prediction = np.asarray(pipeline.predict(test[features]), dtype=float)
        metrics = regression_metrics(test[TARGET], prediction)
        rows.append(
            {
                "model": name,
                "model_id": f"{feature_variant}:{name}",
                "configuration_id": _configuration_id(
                    name, features, clone(base).get_params(deep=False)
                ),
                "feature_variant": feature_variant,
                "test_rows": len(test),
                "historically_exposed": True,
                "selection_use": "none",
                **metrics,
            }
        )
        fitted[name] = pipeline
    return pd.DataFrame(rows), fitted


def score_variant(
    model_id: str,
    pipeline: Any,
    test: pd.DataFrame,
    features: list[str],
) -> tuple[dict[str, Any], pd.DataFrame]:
    prediction = np.asarray(pipeline.predict(test[features]), dtype=float)
    metrics = regression_metrics(test[TARGET], prediction)
    actual = test[TARGET].to_numpy(dtype=float)
    absolute = np.abs(actual - prediction)
    q90 = float(np.quantile(absolute, 0.9))
    detail = test.copy()
    detail["record_offset"] = np.arange(len(test))
    detail["model_id"] = model_id
    detail["feature_variant"] = "top2" if len(features) == 2 else "full"
    detail["predicted_salary_usd"] = prediction
    detail["residual_usd"] = actual - prediction
    detail["absolute_error_usd"] = absolute
    return (
        {
            "model_id": model_id,
            "feature_variant": detail["feature_variant"].iloc[0],
            "evaluation": "historical_test",
            "row_count": len(test),
            **metrics,
            "q90_abs_error_usd": q90,
            "q90_basis": "historical_test_absolute_errors",
            "q90_quantile": 0.9,
            "q90_method": "linear",
            "coverage": float((absolute <= q90).mean()),
        },
        detail,
    )


def permutation_evidence(
    model_id: str,
    pipeline: Any,
    test: pd.DataFrame,
    features: list[str],
    *,
    seed: int,
    repeats: int = 12,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = permutation_importance(
        pipeline,
        test[features],
        test[TARGET],
        scoring="neg_mean_absolute_error",
        n_repeats=repeats,
        random_state=seed,
        n_jobs=1,
    )
    raw = pd.DataFrame(
        {
            "model_id": model_id,
            "raw_feature": features,
            "mae_increase_usd": result.importances_mean,
            "std_usd": result.importances_std,
            "repeats": repeats,
            "seed": seed,
        }
    ).sort_values("mae_increase_usd", ascending=False)
    encoded = extract_encoded_importance(pipeline).rename(columns={"importance": "importance"})
    encoded.insert(0, "model_id", model_id)
    encoded["raw_family"] = encoded["encoded_feature"].map(
        lambda value: _raw_family(str(value), features)
    )
    return raw.reset_index(drop=True), encoded


def select_benchmark_offsets(
    test: pd.DataFrame, eligible: pd.Series, *, limit: int = 3
) -> list[int]:
    if len(test) != len(eligible):
        raise ValueError("eligible mask length must match test rows")
    indices = [int(index) for index in np.flatnonzero(eligible.to_numpy(dtype=bool))]
    if len(indices) <= limit:
        return indices
    return random.Random(2026).sample(indices, limit)
