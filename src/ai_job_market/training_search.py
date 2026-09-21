"""Bounded, MAE-ranked Random Forest search for training-validation/v1."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

from .core import TARGET, candidate_models, make_model_pipeline
from .training_partitions import FoldDeclaration, temporal_fold_splitter


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


def _grid(policy: dict[str, Any]) -> tuple[dict[str, list[Any]], dict[str, Any]]:
    required = {"n_estimators", "max_depth", "min_samples_leaf", "max_features"}
    if required - set(policy):
        raise ValueError("GridSearchCV policy is missing required Random Forest parameters")
    n_estimators = [int(value) for value in policy["n_estimators"]]
    max_depth = [None if value is None else int(value) for value in policy["max_depth"]]
    if not n_estimators or not max_depth or any(value < 1 for value in n_estimators):
        raise ValueError("GridSearchCV parameter grid is invalid")
    return (
        {"model__n_estimators": n_estimators, "model__max_depth": max_depth},
        {
            "min_samples_leaf": int(policy["min_samples_leaf"]),
            "max_features": float(policy["max_features"]),
        },
    )


def gridsearch_rf_search(
    *,
    search_id: str,
    frame: pd.DataFrame,
    membership: pd.DataFrame,
    features: list[str],
    folds: list[FoldDeclaration],
    policy: dict[str, Any],
    seed: int,
    on_step: Callable[[dict[str, Any]], None] | None = None,
) -> SearchResult:
    """Tune only Random Forest hyperparameters on declared temporal inner folds."""
    splitter, positions = temporal_fold_splitter(membership, folds)
    param_grid, fixed = _grid(policy)
    pipeline = make_model_pipeline(
        RandomForestRegressor(random_state=seed, n_jobs=1, **fixed), features
    )
    search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        scoring={"neg_mae": "neg_mean_absolute_error", "r2": "r2"},
        refit=False,
        cv=splitter,
        n_jobs=1,
        error_score=np.nan,
        return_train_score=False,
    )
    search.fit(frame.iloc[positions][features], frame.iloc[positions][TARGET])
    results = search.cv_results_
    rows: list[dict[str, Any]] = []
    for ordinal in range(len(results["params"])):
        params = results["params"][ordinal]
        mae = -float(results["mean_test_neg_mae"][ordinal])
        mae_sd = float(results["std_test_neg_mae"][ordinal])
        raw_r2 = float(results["mean_test_r2"][ordinal])
        raw_r2_sd = float(results["std_test_r2"][ordinal])
        r2 = raw_r2 if np.isfinite(raw_r2) else None
        r2_sd = raw_r2_sd if np.isfinite(raw_r2_sd) else None
        fold_mae = [
            -float(results[f"split{index}_test_neg_mae"][ordinal]) for index in range(len(folds))
        ]
        fold_r2 = [
            value if np.isfinite(value) else None
            for index in range(len(folds))
            for value in [float(results[f"split{index}_test_r2"][ordinal])]
        ]
        status = "completed" if np.isfinite(mae) else "failed"
        full_params = {
            "n_estimators": int(params["model__n_estimators"]),
            "max_depth": params["model__max_depth"],
            **fixed,
        }
        row = {
            "search_id": search_id,
            "method_version": "gridsearchcv-temporal/v1",
            "stage": "grid-search",
            "slot": ordinal + 1,
            "trial_ordinal": ordinal + 1,
            "trial_id": f"{search_id}:candidate-{ordinal + 1}",
            "configuration_id": hashlib.sha256(
                json.dumps(full_params, sort_keys=True).encode()
            ).hexdigest(),
            "params": full_params,
            "n_estimators": full_params["n_estimators"],
            "max_depth": full_params["max_depth"],
            "min_samples_leaf": fixed["min_samples_leaf"],
            "max_features": fixed["max_features"],
            "seed": seed,
            "fold_ids": [fold.fold_id for fold in folds],
            "inner_fold_ids": [fold.fold_id for fold in folds],
            "fold_count": len(folds),
            "fold_mae": fold_mae,
            "fold_r2": fold_r2,
            "mae_mean": mae,
            "mae_sd": mae_sd,
            "r2_mean": r2,
            "r2_sd": r2_sd,
            "r2_reason": None if r2 is not None else "nonfinite_gridsearch_r2",
            "mean_fit_time_s": float(results["mean_fit_time"][ordinal]),
            "std_fit_time_s": float(results["std_fit_time"][ordinal]),
            "ranking_metric": "MAE",
            "ranking_direction": "lower",
            "scoring": "neg_mean_absolute_error",
            "rank": int(results["rank_test_neg_mae"][ordinal]),
            "status": status,
            "failure_reason": None if status == "completed" else "nonfinite_gridsearch_score",
            "evidence_ref": f"tuning_trials.csv#trial_id={search_id}:candidate-{ordinal + 1}",
        }
        rows.append(row)
        if on_step is not None:
            on_step(row)
    trials = pd.DataFrame(rows)
    valid = trials[trials["status"] == "completed"].sort_values(["rank", "trial_ordinal"])
    if valid.empty:
        return SearchResult(
            search_id,
            "failed",
            "all_gridsearch_candidates_failed",
            None,
            trials,
            len(trials) * len(folds),
        )
    return SearchResult(
        search_id,
        "completed",
        None,
        dict(valid.iloc[0]["params"]),
        trials,
        len(trials) * len(folds),
    )


def conditional_gridsearch_rf_search(
    *,
    selected_family: str,
    **kwargs: Any,
) -> SearchResult:
    if selected_family != "Random Forest":
        return SearchResult(
            kwargs["search_id"],
            "skipped",
            "selected_family_is_not_random_forest",
            None,
            pd.DataFrame(),
            0,
        )
    return gridsearch_rf_search(**kwargs)


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
