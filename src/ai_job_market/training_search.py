"""Bounded, MAE-ranked Random Forest search for training-validation/v1."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor

from .core import TARGET, candidate_models, make_model_pipeline

EvaluateTrial = Callable[[dict[str, Any], list[str]], list[float]]


@dataclass
class FinalVariantFits:
    pipelines: dict[str, Any]
    fit_rows: int
    train_row_ids: tuple[str, ...]
    params: dict[str, Any]


@dataclass
class SearchResult:
    search_id: str
    status: str
    reason: str | None
    winner: dict[str, Any] | None
    trials: pd.DataFrame
    actual_fit_count: int


def _config_key(params: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(params, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _rank_key(row: dict[str, Any]) -> tuple[Any, ...]:
    params = row["params"]
    depth = params["max_depth"]
    return (
        row["mae_mean"],
        params["n_estimators"],
        float("inf") if depth is None else depth,
        -params["min_samples_leaf"],
        params["max_features"],
        row["trial_ordinal"],
    )


def stepwise_rf_search(
    *,
    search_id: str,
    fold_ids: list[str],
    policy: dict[str, Any],
    evaluate: EvaluateTrial,
    seed: int,
) -> SearchResult:
    if len(fold_ids) != 3 or len(set(fold_ids)) != 3:
        raise ValueError("Random Forest search requires exactly three distinct inner folds")
    stage_definitions = [
        ("initial-grid", None, policy["anchors"]),
        ("n-estimators", "n_estimators", policy["n_estimators"]),
        ("max-depth", "max_depth", policy["max_depth"]),
        ("min-samples-leaf", "min_samples_leaf", policy["min_samples_leaf"]),
        ("max-features", "max_features", policy["max_features"]),
    ]
    cache: dict[str, tuple[list[float], str]] = {}
    rows: list[dict[str, Any]] = []
    incumbent: dict[str, Any] | None = None
    ordinal = 0
    actual_fit_count = 0
    for stage, parameter, values in stage_definitions:
        stage_rows: list[dict[str, Any]] = []
        for slot, value in enumerate(values, start=1):
            ordinal += 1
            if parameter is None:
                params = dict(value)
            else:
                if incumbent is None:
                    raise ValueError("search incumbent is unavailable")
                params = incumbent | {parameter: value}
            params["n_estimators"] = int(params["n_estimators"])
            params["min_samples_leaf"] = int(params["min_samples_leaf"])
            params["max_features"] = float(params["max_features"])
            if params["max_depth"] is not None:
                params["max_depth"] = int(params["max_depth"])
            key = _config_key(params)
            reused_from = None
            search_wall_start = time.perf_counter()
            search_cpu_start = time.process_time()
            if key in cache:
                scores, reused_from = cache[key]
                status = "reused"
            else:
                scores = [float(score) for score in evaluate(params.copy(), list(fold_ids))]
                if len(scores) != len(fold_ids) or not np.isfinite(scores).all():
                    raise ValueError("tuning trial must return one finite MAE per inner fold")
                cache[key] = (scores, f"{search_id}:trial-{ordinal}")
                status = "completed"
                actual_fit_count += len(fold_ids)
            row = {
                "search_id": search_id,
                "stage": stage,
                "slot": slot,
                "trial_ordinal": ordinal,
                "trial_id": f"{search_id}:trial-{ordinal}",
                "configuration_id": key,
                "params": params,
                "seed": seed,
                "fold_ids": list(fold_ids),
                "fold_count": len(fold_ids),
                "fold_mae": list(scores),
                "mae_mean": float(np.mean(scores)),
                "mae_sd": float(np.std(scores, ddof=0)),
                "ranking_metric": "MAE",
                "ranking_direction": "lower",
                "status": status,
                "reused_from": reused_from,
                "search_wall_s": time.perf_counter() - search_wall_start,
                "search_cpu_s": time.process_time() - search_cpu_start,
            }
            rows.append(row)
            stage_rows.append(row)
        incumbent = dict(min(stage_rows, key=_rank_key)["params"])
    return SearchResult(
        search_id=search_id,
        status="completed",
        reason=None,
        winner=incumbent,
        trials=pd.DataFrame(rows),
        actual_fit_count=actual_fit_count,
    )


def conditional_stepwise_rf_search(
    *,
    selected_family: str,
    search_id: str,
    fold_ids: list[str],
    policy: dict[str, Any],
    evaluate: EvaluateTrial,
    seed: int,
) -> SearchResult:
    if selected_family != "Random Forest":
        return SearchResult(
            search_id=search_id,
            status="skipped",
            reason="selected_family_is_not_random_forest",
            winner=None,
            trials=pd.DataFrame(),
            actual_fit_count=0,
        )
    return stepwise_rf_search(
        search_id=search_id,
        fold_ids=fold_ids,
        policy=policy,
        evaluate=evaluate,
        seed=seed,
    )


def run_declared_rf_searches(
    *,
    selected_family: str,
    contexts: dict[str, list[str]],
    policy: dict[str, Any],
    evaluator_factory: Callable[[str], EvaluateTrial],
    seed: int,
) -> dict[str, SearchResult]:
    expected = {f"outer-{index}" for index in range(1, 6)} | {"final-train"}
    if set(contexts) != expected:
        raise ValueError("nested tuning requires five outer contexts and one final TRAIN context")
    return {
        context_id: conditional_stepwise_rf_search(
            selected_family=selected_family,
            search_id=f"rf-{context_id}",
            fold_ids=fold_ids,
            policy=policy,
            evaluate=evaluator_factory(context_id),
            seed=seed,
        )
        for context_id, fold_ids in contexts.items()
    }


def fit_final_variants(
    frame: pd.DataFrame,
    membership: pd.DataFrame,
    *,
    selected_family: str,
    params: dict[str, Any],
    full_features: list[str],
    top2_features: list[str],
    seed: int,
) -> FinalVariantFits:
    if top2_features != ["job_category", "years_of_experience"]:
        raise ValueError("Top-2 feature order does not match the frozen contract")
    train = membership[membership["partition"] == "TRAIN"]
    if train.empty:
        raise ValueError("final fitting requires a nonempty TRAIN population")
    positions = train["source_position"].to_numpy(dtype=int)
    if selected_family == "Random Forest":
        estimator = RandomForestRegressor(random_state=seed, n_jobs=1, **params)
    else:
        factories = candidate_models(seed)
        if selected_family not in factories:
            raise ValueError("selected family is not a declared candidate")
        estimator = clone(factories[selected_family])
        actual = estimator.get_params(deep=False)
        for name, value in params.items():
            if actual.get(name) != value:
                raise ValueError("provided final parameters do not match frozen family defaults")
    pipelines: dict[str, Any] = {}
    for role, features in (("final_full", full_features), ("final_top2", top2_features)):
        pipeline = make_model_pipeline(clone(estimator), list(features))
        pipeline.fit(frame.iloc[positions][features], frame.iloc[positions][TARGET])
        pipelines[role] = pipeline
    return FinalVariantFits(
        pipelines=pipelines,
        fit_rows=len(positions),
        train_row_ids=tuple(train["row_id"]),
        params=dict(params),
    )


def evaluate_matched_variants(
    *,
    fold_winners: dict[str, dict[str, Any]],
    full_features: list[str],
    top2_features: list[str],
    evaluate: Callable[[str, list[str], dict[str, Any], str], dict[str, Any]],
) -> list[dict[str, Any]]:
    if top2_features != ["job_category", "years_of_experience"]:
        raise ValueError("Top-2 feature order does not match the frozen contract")
    if not set(top2_features) < set(full_features):
        raise ValueError("Top-2 features must be a strict subset of full features")
    rows: list[dict[str, Any]] = []
    for fold_id, params in fold_winners.items():
        for variant, features in (("full", full_features), ("top2", top2_features)):
            evidence = evaluate(variant, list(features), dict(params), fold_id)
            rows.append(
                {
                    "fold_id": fold_id,
                    "variant": variant,
                    "features": list(features),
                    "params": dict(params),
                    **evidence,
                }
            )
    return rows
