"""Explicit-fold regression evaluation for training-validation/v1."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score

from .core import TARGET, make_model_pipeline
from .training_partitions import FoldDeclaration

SIMPLICITY_ORDER = (
    "Dummy Median",
    "Linear Regression",
    "Ridge Regression",
    "Gradient Boosting",
    "Random Forest",
)


@dataclass
class EvaluationResult:
    fold_metrics: pd.DataFrame
    oof_predictions: pd.DataFrame
    fitted_pipelines: dict[tuple[str, str], Any]


def regression_metrics_strict(y_true, y_pred) -> dict[str, float | str | None]:
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    if len(actual) != len(predicted) or len(actual) == 0:
        raise ValueError("actual and predicted values must be nonempty and aligned")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("actual and predicted values must be finite")
    values: dict[str, float | str | None] = {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(mean_squared_error(actual, predicted) ** 0.5),
        "MedAE": float(median_absolute_error(actual, predicted)),
        "R2": None,
        "R2_reason": None,
    }
    if len(actual) < 2:
        values["R2_reason"] = "fewer_than_two_rows"
    elif np.all(actual == actual[0]):
        values["R2_reason"] = "constant_target"
    else:
        values["R2"] = float(r2_score(actual, predicted, force_finite=False))
    return values


def _configuration_id(model_name: str, features: list[str], model: BaseEstimator) -> str:
    material = json.dumps(
        {
            "model": model_name,
            "features": features,
            "params": model.get_params(deep=False),
        },
        sort_keys=True,
        default=str,
        separators=(",", ":"),
    )
    return hashlib.sha256(material.encode()).hexdigest()


def _positions(membership: pd.DataFrame, row_ids: tuple[str, ...]) -> np.ndarray:
    lookup = membership.set_index("row_id")["source_position"]
    try:
        return lookup.loc[list(row_ids)].to_numpy(dtype=int)
    except KeyError as error:
        raise ValueError("fold row identity is absent from dataset membership") from error


def evaluate_frozen_models(
    frame: pd.DataFrame,
    membership: pd.DataFrame,
    features: list[str],
    models: dict[str, BaseEstimator],
    folds: list[FoldDeclaration],
) -> EvaluationResult:
    """Fit each frozen model once per explicit fold and retain auditable OOF rows."""
    required = set(features) | {TARGET}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"training frame is missing columns: {missing}")
    metrics_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    fitted: dict[tuple[str, str], Any] = {}
    for model_name, base_model in models.items():
        configuration_id = _configuration_id(model_name, features, base_model)
        for fold in folds:
            train_positions = _positions(membership, fold.train_row_ids)
            validation_positions = _positions(membership, fold.validation_row_ids)
            pipeline = make_model_pipeline(clone(base_model), features)
            started = time.perf_counter()
            cpu_started = time.process_time()
            pipeline.fit(frame.iloc[train_positions][features], frame.iloc[train_positions][TARGET])
            fit_cpu_s = time.process_time() - cpu_started
            fit_wall_s = time.perf_counter() - started
            started = time.perf_counter()
            cpu_started = time.process_time()
            train_prediction = np.asarray(
                pipeline.predict(frame.iloc[train_positions][features]), dtype=float
            )
            train_predict_cpu_s = time.process_time() - cpu_started
            train_predict_wall_s = time.perf_counter() - started
            started = time.perf_counter()
            cpu_started = time.process_time()
            validation_prediction = np.asarray(
                pipeline.predict(frame.iloc[validation_positions][features]), dtype=float
            )
            validation_predict_cpu_s = time.process_time() - cpu_started
            validation_predict_wall_s = time.perf_counter() - started
            if not np.isfinite(train_prediction).all() or not np.isfinite(
                validation_prediction
            ).all():
                raise ValueError(f"model {model_name} produced non-finite predictions")
            train_metrics = regression_metrics_strict(
                frame.iloc[train_positions][TARGET], train_prediction
            )
            validation_metrics = regression_metrics_strict(
                frame.iloc[validation_positions][TARGET], validation_prediction
            )
            encoded_feature_count = len(
                pipeline.named_steps["preprocess"].get_feature_names_out()
            )
            metrics_rows.append(
                {
                    "model": model_name,
                    "configuration_id": configuration_id,
                    "fold_id": fold.fold_id,
                    "fold_ordinal": fold.ordinal,
                    "train_rows": len(train_positions),
                    "validation_rows": len(validation_positions),
                    "train_months": list(fold.train_months),
                    "validation_months": list(fold.validation_months),
                    "encoded_feature_count": encoded_feature_count,
                    "fit_wall_s": fit_wall_s,
                    "fit_cpu_s": fit_cpu_s,
                    "train_predict_wall_s": train_predict_wall_s,
                    "train_predict_cpu_s": train_predict_cpu_s,
                    "validation_predict_wall_s": validation_predict_wall_s,
                    "validation_predict_cpu_s": validation_predict_cpu_s,
                    **{f"train_{key}": value for key, value in train_metrics.items()},
                    **{
                        f"validation_{key}": value
                        for key, value in validation_metrics.items()
                    },
                }
            )
            actual = frame.iloc[validation_positions][TARGET].to_numpy(dtype=float)
            for row_id, y_true, y_pred in zip(
                fold.validation_row_ids, actual, validation_prediction, strict=True
            ):
                prediction_rows.append(
                    {
                        "model": model_name,
                        "configuration_id": configuration_id,
                        "fold_id": fold.fold_id,
                        "row_id": row_id,
                        "partition": "OUTER_VALIDATION",
                        "actual": float(y_true),
                        "predicted": float(y_pred),
                        "residual": float(y_true - y_pred),
                        "absolute_error": float(abs(y_true - y_pred)),
                    }
                )
            fitted[(model_name, fold.fold_id)] = pipeline
    return EvaluationResult(
        fold_metrics=pd.DataFrame(metrics_rows),
        oof_predictions=pd.DataFrame(prediction_rows),
        fitted_pipelines=fitted,
    )


def summarize_candidates(
    fold_metrics: pd.DataFrame, oof_predictions: pd.DataFrame
) -> pd.DataFrame:
    summaries: list[dict[str, Any]] = []
    for model_name, rows in fold_metrics.groupby("model", sort=False):
        oof = oof_predictions[oof_predictions["model"] == model_name]
        pooled = regression_metrics_strict(oof["actual"], oof["predicted"])
        mae = rows["validation_MAE"].astype(float)
        worst_index = mae.idxmax()
        valid_r2 = rows["validation_R2"].dropna().astype(float)
        summaries.append(
            {
                "model": model_name,
                "configuration_id": rows.iloc[0]["configuration_id"],
                "valid_fold_count": int(len(rows)),
                "failed_fold_count": 0,
                "cv_mae_mean_usd": float(mae.mean()),
                "cv_mae_sd_usd": float(mae.std(ddof=0)),
                "cv_mae_min_usd": float(mae.min()),
                "cv_mae_max_usd": float(mae.max()),
                "cv_rmse_mean_usd": float(rows["validation_RMSE"].mean()),
                "cv_medae_mean_usd": float(rows["validation_MedAE"].mean()),
                "cv_r2_mean": float(valid_r2.mean()) if len(valid_r2) == len(rows) else None,
                "cv_r2_valid_folds": int(len(valid_r2)),
                "worst_fold_id": rows.loc[worst_index, "fold_id"],
                "pooled_oof_mae_usd": pooled["MAE"],
                "pooled_oof_rmse_usd": pooled["RMSE"],
                "pooled_oof_medae_usd": pooled["MedAE"],
                "pooled_oof_r2": pooled["R2"],
                "total_cv_fit_wall_s": float(rows["fit_wall_s"].sum()),
                "median_fold_fit_wall_s": float(rows["fit_wall_s"].median()),
            }
        )
    return pd.DataFrame(summaries).sort_values("cv_mae_mean_usd").reset_index(drop=True)


def derive_performance_comparison(
    oof_predictions: pd.DataFrame,
    *,
    variant_fold_metrics: pd.DataFrame,
    tuning_baseline_mae: float | None,
    tuning_selected_mae: float | None,
) -> dict[str, Any]:
    required = {"model", "residual", "absolute_error"}
    if required - set(oof_predictions.columns):
        raise ValueError("OOF predictions are missing derived-performance fields")
    dummy = oof_predictions[oof_predictions["model"] == "Dummy Median"]
    if dummy.empty:
        raise ValueError("Dummy Median evidence is required for skill scores")
    dummy_mae = float(dummy["absolute_error"].mean())
    rows: list[dict[str, Any]] = []
    for model, group in oof_predictions.groupby("model", sort=False):
        mae = float(group["absolute_error"].mean())
        rows.append(
            {
                "model": model,
                "mean_residual_usd": float(group["residual"].mean()),
                "p90_absolute_error_usd": float(
                    np.quantile(group["absolute_error"], 0.9, method="linear")
                ),
                "dummy_skill_mae": None if dummy_mae == 0 else 1 - mae / dummy_mae,
                "dummy_skill_reason": "zero_dummy_mae" if dummy_mae == 0 else None,
            }
        )
    variant_required = {"fold_id", "variant", "validation_MAE"}
    if variant_required - set(variant_fold_metrics.columns):
        raise ValueError("variant evidence is incomplete")
    paired = variant_fold_metrics.pivot(
        index="fold_id", columns="variant", values="validation_MAE"
    )
    if set(paired.columns) != {"full", "top2"} or paired.isna().any().any():
        raise ValueError("Full and Top-2 evidence must be paired by fold")
    tuning_delta = (
        None
        if tuning_baseline_mae is None or tuning_selected_mae is None
        else float(tuning_selected_mae - tuning_baseline_mae)
    )
    return {
        "models": pd.DataFrame(rows),
        "dummy_mae_usd": dummy_mae,
        "top2_mae_delta_usd": float((paired["top2"] - paired["full"]).mean()),
        "tuning_mae_delta_usd": tuning_delta,
    }


def fit_diagnostics(fold_metrics: pd.DataFrame, *, gap_threshold: float) -> pd.DataFrame:
    dummy = fold_metrics[fold_metrics["model"] == "Dummy Median"].set_index("fold_id")
    if len(dummy) == 0:
        raise ValueError("fit diagnostics require matched Dummy Median folds")
    rows: list[dict[str, Any]] = []
    for _, metric in fold_metrics.iterrows():
        fold_id = metric["fold_id"]
        if fold_id not in dummy.index:
            raise ValueError("fit diagnostics require matched Dummy Median folds")
        if metric["model"] == "Dummy Median":
            indication = "baseline"
            train_gain = validation_gain = normalized_gap = None
        else:
            dummy_train = float(dummy.loc[fold_id, "train_MAE"])
            dummy_validation = float(dummy.loc[fold_id, "validation_MAE"])
            train_gain = None if dummy_train == 0 else 1 - float(metric.train_MAE) / dummy_train
            validation_gain = (
                None
                if dummy_validation == 0
                else 1 - float(metric.validation_MAE) / dummy_validation
            )
            normalized_gap = (
                None
                if dummy_validation == 0
                else (float(metric.validation_MAE) - float(metric.train_MAE)) / dummy_validation
            )
            validation_r2 = metric.validation_R2
            if train_gain is None or validation_gain is None or normalized_gap is None:
                indication = "inconclusive"
            elif train_gain > 0 and normalized_gap > gap_threshold:
                indication = "overfitting_indication"
            elif train_gain <= 0 and validation_gain <= 0:
                indication = "underfitting_indication"
            elif validation_gain > 0 and validation_r2 is not None and validation_r2 >= 0 and normalized_gap <= gap_threshold:
                indication = "good_fit_indication"
            else:
                indication = "inconclusive"
        rows.append(
            {
                "model": metric["model"],
                "fold_id": fold_id,
                "train_gain_vs_dummy": train_gain,
                "validation_gain_vs_dummy": validation_gain,
                "normalized_mae_gap": normalized_gap,
                "fold_indication": indication,
                "gap_threshold": gap_threshold,
            }
        )
    out = pd.DataFrame(rows)
    aggregate: dict[str, str] = {}
    for model_name, group in out.groupby("model", sort=False):
        if model_name == "Dummy Median":
            aggregate[model_name] = "baseline"
            continue
        counts = group["fold_indication"].value_counts()
        top = str(counts.index[0]) if len(counts) else "inconclusive"
        aggregate[model_name] = top if int(counts.iloc[0]) >= 4 else "mixed_or_inconclusive"
    out["aggregate_indication"] = out["model"].map(aggregate)
    return out


def run_feature_family_ablation(
    frame: pd.DataFrame,
    membership: pd.DataFrame,
    folds: list[FoldDeclaration],
    *,
    features: list[str],
    families: dict[str, list[str]],
    model: BaseEstimator,
    baseline_fold_metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    approved = set(features)
    if set().union(*map(set, families.values())) != approved:
        raise ValueError("feature families must exactly cover the approved feature policy")
    baseline = baseline_fold_metrics.set_index("fold_id")["validation_MAE"]
    summary_rows: list[dict[str, Any]] = []
    fold_tables: list[pd.DataFrame] = []
    for family, removed in families.items():
        remaining = [feature for feature in features if feature not in removed]
        result = evaluate_frozen_models(
            frame, membership, remaining, {"Random Forest": model}, folds
        )
        rows = result.fold_metrics.copy()
        rows["removed_family"] = family
        rows["baseline_validation_MAE"] = rows["fold_id"].map(baseline)
        rows["mae_delta_vs_full"] = rows["validation_MAE"] - rows["baseline_validation_MAE"]
        fold_tables.append(rows)
        summary_rows.append(
            {
                "removed_family": family,
                "removed_features": "|".join(removed),
                "features": "|".join(remaining),
                "feature_count": len(remaining),
                "validation_mae_mean_usd": float(rows["validation_MAE"].mean()),
                "mae_delta_vs_full_mean_usd": float(rows["mae_delta_vs_full"].mean()),
            }
        )
    return pd.DataFrame(summary_rows), pd.concat(fold_tables, ignore_index=True)


def _raw_feature(encoded_name: str, features: list[str]) -> str:
    suffix = encoded_name.split("__", 1)[-1]
    for feature in sorted(features, key=len, reverse=True):
        if suffix == feature or suffix.startswith(f"{feature}_"):
            return feature
    return "unknown"


def extract_rf_fold_importance(
    result: EvaluationResult,
    *,
    features: list[str],
    families: dict[str, list[str]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_to_family = {
        feature: family for family, family_features in families.items() for feature in family_features
    }
    encoded_rows: list[dict[str, Any]] = []
    for (model_name, fold_id), pipeline in result.fitted_pipelines.items():
        if model_name != "Random Forest":
            continue
        estimator = pipeline.named_steps["model"]
        if not hasattr(estimator, "feature_importances_"):
            raise ValueError("Random Forest fit does not expose impurity importance")
        names = pipeline.named_steps["preprocess"].get_feature_names_out()
        values = np.asarray(estimator.feature_importances_, dtype=float)
        if len(names) != len(values):
            raise ValueError("encoded importance does not align with fitted vocabulary")
        for name, value in zip(names, values, strict=True):
            raw = _raw_feature(str(name), features)
            encoded_rows.append(
                {
                    "model": model_name,
                    "fold_id": fold_id,
                    "encoded_feature": str(name),
                    "raw_feature": raw,
                    "family": feature_to_family.get(raw, "unknown"),
                    "present": True,
                    "method": "encoded_impurity",
                    "importance": float(value),
                }
            )
    encoded = pd.DataFrame(encoded_rows)
    if encoded.empty:
        raise ValueError("no existing Random Forest fits were supplied for importance")
    family = (
        encoded.groupby(["model", "fold_id", "family"], as_index=False, sort=False)[
            "importance"
        ].sum()
    )
    expected_folds = [
        fold_id
        for (model_name, fold_id) in result.fitted_pipelines
        if model_name == "Random Forest"
    ]
    full_index = pd.MultiIndex.from_product(
        [["Random Forest"], expected_folds, list(families)],
        names=["model", "fold_id", "family"],
    )
    family = (
        family.set_index(["model", "fold_id", "family"])
        .reindex(full_index, fill_value=0.0)
        .reset_index()
    )
    totals = family.groupby("fold_id")["importance"].transform("sum")
    family["normalized_importance"] = np.where(totals > 0, family["importance"] / totals, 0.0)
    fold_order = (
        result.fold_metrics[result.fold_metrics["model"] == "Random Forest"]
        .sort_values("fold_ordinal")["fold_id"]
        .tolist()
    )
    drift_rows: list[dict[str, Any]] = []
    vectors = {
        fold_id: family[family["fold_id"] == fold_id]
        .set_index("family")["normalized_importance"]
        .reindex(list(families), fill_value=0.0)
        for fold_id in fold_order
    }
    for previous, current in zip(fold_order, fold_order[1:]):
        drift_rows.append(
            {
                "previous_fold_id": previous,
                "fold_id": current,
                "half_l1_drift": float(0.5 * np.abs(vectors[current] - vectors[previous]).sum()),
            }
        )
    return encoded, family, pd.DataFrame(drift_rows)


def extract_fitted_encoded_importance(
    pipeline: Any, *, features: list[str], model_role_id: str
) -> pd.DataFrame:
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    estimator = pipeline.named_steps["model"]
    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype=float)
        method = "encoded_impurity"
    elif hasattr(estimator, "coef_"):
        values = np.abs(np.asarray(estimator.coef_, dtype=float).reshape(-1))
        method = "absolute_coefficient"
    else:
        return pd.DataFrame(
            [
                {
                    "model_role_id": model_role_id,
                    "encoded_feature": str(name),
                    "raw_feature": _raw_feature(str(name), features),
                    "method": "unavailable_no_estimator_importance",
                    "importance": None,
                }
                for name in names
            ]
        )
    if len(names) != len(values) or not np.isfinite(values).all():
        raise ValueError("fitted encoded importance does not align with vocabulary")
    return pd.DataFrame(
        [
            {
                "model_role_id": model_role_id,
                "encoded_feature": str(name),
                "raw_feature": _raw_feature(str(name), features),
                "method": method,
                "importance": float(value),
            }
            for name, value in zip(names, values, strict=True)
        ]
    )


def score_frozen_pipelines(
    frame: pd.DataFrame,
    membership: pd.DataFrame,
    *,
    pipelines: dict[str, Any],
    features_by_model: dict[str, list[str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if set(pipelines) != set(features_by_model):
        raise ValueError("pipeline and feature-role identities differ")
    holdout = membership[membership["partition"] == "EVALUATION_HOLDOUT"]
    if holdout.empty:
        raise ValueError("evaluation holdout is empty")
    positions = holdout["source_position"].to_numpy(dtype=int)
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    actual = frame.iloc[positions][TARGET].to_numpy(dtype=float)
    for role, pipeline in pipelines.items():
        features = features_by_model[role]
        started = time.perf_counter()
        cpu_started = time.process_time()
        predicted = np.asarray(pipeline.predict(frame.iloc[positions][features]), dtype=float)
        predict_cpu_s = time.process_time() - cpu_started
        predict_wall_s = time.perf_counter() - started
        metrics = regression_metrics_strict(actual, predicted)
        metric_rows.append(
            {
                "model_role_id": role,
                "partition": "EVALUATION_HOLDOUT",
                "rows": len(positions),
                "predict_wall_s": predict_wall_s,
                "predict_cpu_s": predict_cpu_s,
                **metrics,
            }
        )
        source = frame.iloc[positions]
        for row_id, y_true, y_pred, (_, source_row) in zip(
            holdout["row_id"], actual, predicted, source.iterrows(), strict=True
        ):
            prediction_rows.append(
                {
                    "model_role_id": role,
                    "row_id": row_id,
                    "partition": "EVALUATION_HOLDOUT",
                    "actual": float(y_true),
                    "predicted": float(y_pred),
                    "residual": float(y_true - y_pred),
                    "absolute_error": float(abs(y_true - y_pred)),
                    "job_category": source_row.get("job_category"),
                    "country": source_row.get("country"),
                    "years_of_experience": float(source_row.get("years_of_experience")),
                }
            )
    return pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows)


def permutation_mae_importance(
    pipeline: Any,
    evaluation_frame: pd.DataFrame,
    *,
    features: list[str],
    families: dict[str, list[str]],
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    if repeats <= 0:
        raise ValueError("permutation repeats must be positive")
    actual = evaluation_frame[TARGET].to_numpy(dtype=float)
    baseline = np.asarray(pipeline.predict(evaluation_frame[features]), dtype=float)
    baseline_mae = float(mean_absolute_error(actual, baseline))
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    groups = [("raw_feature", feature, [feature]) for feature in features]
    groups += [("joint_family", family, columns) for family, columns in families.items()]
    for kind, name, columns in groups:
        if not set(columns) <= set(features):
            raise ValueError("permutation group is outside the fitted feature policy")
        for repeat in range(1, repeats + 1):
            order = rng.permutation(len(evaluation_frame))
            permuted = evaluation_frame[features].copy()
            for column in columns:
                permuted[column] = evaluation_frame[column].to_numpy()[order]
            predicted = np.asarray(pipeline.predict(permuted), dtype=float)
            permuted_mae = float(mean_absolute_error(actual, predicted))
            rows.append(
                {
                    "kind": kind,
                    "name": name,
                    "repeat": repeat,
                    "scoring": "mae_increase_usd",
                    "baseline_mae_usd": baseline_mae,
                    "permuted_mae_usd": permuted_mae,
                    "mae_increase": permuted_mae - baseline_mae,
                    "seed": seed,
                }
            )
    return pd.DataFrame(rows)


def subgroup_metrics(predictions: pd.DataFrame, *, small_support: int = 20) -> pd.DataFrame:
    required = {
        "model_role_id",
        "actual",
        "predicted",
        "job_category",
        "country",
        "years_of_experience",
    }
    if required - set(predictions.columns):
        raise ValueError("prediction evidence is missing subgroup columns")
    working = predictions.copy()
    working["experience_bucket"] = pd.cut(
        working["years_of_experience"],
        bins=[0, 3, 7, np.inf],
        right=False,
        labels=["[0,3)", "[3,7)", "[7,+inf)"],
    )
    if working["experience_bucket"].isna().any():
        raise ValueError("years_of_experience is outside subgroup bounds")
    rows: list[dict[str, Any]] = []
    for model_role, model_rows in working.groupby("model_role_id", sort=False):
        for subgroup in ("job_category", "country", "experience_bucket"):
            for value, group in model_rows.groupby(subgroup, observed=True, dropna=False):
                metrics = regression_metrics_strict(group["actual"], group["predicted"])
                rows.append(
                    {
                        "model_role_id": model_role,
                        "subgroup": subgroup,
                        "value": str(value),
                        "support": len(group),
                        "small_sample": len(group) < small_support,
                        **metrics,
                    }
                )
    return pd.DataFrame(rows)


def uncertainty_evidence(
    predictions: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    roles = predictions["model_role_id"].unique()
    if len(roles) != 1:
        raise ValueError("uncertainty evidence must be computed for one model role at a time")
    errors = predictions["absolute_error"].to_numpy(dtype=float)
    if len(errors) == 0 or not np.isfinite(errors).all():
        raise ValueError("uncertainty evidence requires finite holdout absolute errors")
    q90 = float(np.quantile(errors, 0.9, method="linear"))
    with_bands = predictions.copy()
    with_bands["lower"] = with_bands["predicted"] - q90
    with_bands["upper"] = with_bands["predicted"] + q90
    coverage = float(
        ((with_bands["actual"] >= with_bands["lower"]) & (with_bands["actual"] <= with_bands["upper"])).mean()
    )
    return (
        {
            "model_role_id": str(roles[0]),
            "q90_absolute_error_usd": q90,
            "quantile": 0.9,
            "quantile_method": "linear",
            "same_population_coverage": coverage,
            "basis": "same_holdout_descriptive_not_calibrated",
            "rows": len(with_bands),
        },
        with_bands,
    )


def select_family(summary: pd.DataFrame) -> dict[str, Any]:
    expected = set(SIMPLICITY_ORDER)
    if set(summary["model"]) != expected:
        raise ValueError("family selection requires exactly the five declared candidate models")
    if summary[["cv_mae_mean_usd", "cv_mae_sd_usd"]].isna().any().any():
        raise ValueError("family selection requires complete finite MAE evidence")
    best = summary.sort_values(["cv_mae_mean_usd", "model"]).iloc[0]
    best_low = float(best.cv_mae_mean_usd - best.cv_mae_sd_usd)
    best_high = float(best.cv_mae_mean_usd + best.cv_mae_sd_usd)
    overlap: list[str] = []
    for _, candidate in summary.iterrows():
        low = float(candidate.cv_mae_mean_usd - candidate.cv_mae_sd_usd)
        high = float(candidate.cv_mae_mean_usd + candidate.cv_mae_sd_usd)
        if low <= best_high and best_low <= high:
            overlap.append(str(candidate.model))
    selected = next(model for model in SIMPLICITY_ORDER if model in overlap)
    return {
        "selection_method": "lowest_mean_mae_then_sd_overlap_simplicity_v1",
        "lowest_mae_model": str(best.model),
        "overlap_models": [model for model in SIMPLICITY_ORDER if model in overlap],
        "selected_family": selected,
        "runtime_used_for_selection": False,
    }
