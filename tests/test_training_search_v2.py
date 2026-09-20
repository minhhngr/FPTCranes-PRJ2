from __future__ import annotations

import pandas as pd

from ai_job_market.core import MODEL_FEATURES, TARGET
from ai_job_market.training_partitions import build_partition_membership
from ai_job_market.training_search import (
    conditional_stepwise_rf_search,
    evaluate_matched_variants,
    fit_final_variants,
    run_declared_rf_searches,
    stepwise_rf_search,
)


def search_policy() -> dict:
    return {
        "anchors": [
            {"n_estimators": 300, "min_samples_leaf": 2, "max_features": 0.7, "max_depth": 20},
            {"n_estimators": 100, "min_samples_leaf": 2, "max_features": 0.8, "max_depth": None},
            {"n_estimators": 80, "min_samples_leaf": 1, "max_features": 0.8, "max_depth": None},
            {"n_estimators": 200, "min_samples_leaf": 1, "max_features": 0.7, "max_depth": 20},
        ],
        "n_estimators": [50, 100, 150, 200, 250, 300],
        "max_depth": [10, 15, 20, 25, 30, None],
        "min_samples_leaf": [1, 2, 4, 8],
        "max_features": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    }


def objective(params: dict, fold_ids: list[str]) -> list[float]:
    depth = 40 if params["max_depth"] is None else params["max_depth"]
    base = (
        abs(params["n_estimators"] - 150) / 100
        + abs(depth - 15) / 10
        + abs(params["min_samples_leaf"] - 4)
        + abs(params["max_features"] - 0.6) * 10
    )
    return [base + index / 100 for index, _ in enumerate(fold_ids)]


def test_stepwise_search_has_26_slots_three_folds_and_carries_winners() -> None:
    result = stepwise_rf_search(
        search_id="search-1",
        fold_ids=["inner-1", "inner-2", "inner-3"],
        policy=search_policy(),
        evaluate=objective,
        seed=42,
    )
    assert len(result.trials) == 26
    assert result.trials["fold_count"].eq(3).all()
    assert result.trials["ranking_metric"].eq("MAE").all()
    assert result.winner == {
        "n_estimators": 150,
        "min_samples_leaf": 4,
        "max_features": 0.6,
        "max_depth": 15,
    }
    assert result.trials["status"].isin(["completed", "reused"]).all()
    assert result.actual_fit_count <= 26 * 3
    assert result.trials.groupby("stage")["slot"].count().tolist() == [4, 6, 6, 4, 6]


def test_mae_ties_use_declared_parameter_simplicity() -> None:
    result = stepwise_rf_search(
        search_id="ties",
        fold_ids=["a", "b", "c"],
        policy=search_policy(),
        evaluate=lambda params, folds: [1.0, 1.0, 1.0],
        seed=42,
    )
    assert result.winner["n_estimators"] == 50
    assert result.winner["max_depth"] == 10
    assert result.winner["min_samples_leaf"] == 8
    assert result.winner["max_features"] == 0.5


def test_conditional_search_skips_when_random_forest_is_not_selected() -> None:
    result = conditional_stepwise_rf_search(
        selected_family="Linear Regression",
        search_id="skip",
        fold_ids=["a", "b", "c"],
        policy=search_policy(),
        evaluate=objective,
        seed=42,
    )
    assert result.status == "skipped"
    assert result.reason == "selected_family_is_not_random_forest"
    assert len(result.trials) == 0


def test_six_declared_searches_keep_outer_contexts_independent() -> None:
    calls = []

    def factory(context_id: str):
        def evaluate(params, fold_ids):
            calls.append((context_id, tuple(fold_ids), params.copy()))
            return objective(params, fold_ids)

        return evaluate

    contexts = {
        **{f"outer-{index}": [f"o{index}-i{fold}" for fold in range(1, 4)] for index in range(1, 6)},
        "final-train": ["final-i1", "final-i2", "final-i3"],
    }
    results = run_declared_rf_searches(
        selected_family="Random Forest",
        contexts=contexts,
        policy=search_policy(),
        evaluator_factory=factory,
        seed=42,
    )
    assert set(results) == set(contexts)
    assert all(len(result.trials) == 26 for result in results.values())
    assert all(len({tuple(call[1]) for call in calls if call[0] == context}) == 1 for context in contexts)


def test_final_variants_fit_train_only_with_common_parameters() -> None:
    rows = []
    for month in range(1, 13):
        for item in range(2):
            rows.append(
                {
                    "posting_year": 2025,
                    "posting_month": month,
                    "job_title": "Data Scientist",
                    "job_category": "Data Science",
                    "years_of_experience": item + month / 10,
                    "education_required": "Bachelor's",
                    "city": "Hanoi",
                    "country": "Vietnam",
                    "remote_work": "Hybrid",
                    "company_size": "Mid-size (501-5000)",
                    "industry": "Technology",
                    "demand_score": 50 + month,
                    "benefits_score_10": 7.0,
                    "required_skills": "Python|SQL",
                    "skill_count": 2,
                    TARGET: 60000 + month * 1000 + item * 3000,
                }
            )
    frame = pd.DataFrame(rows)
    membership = build_partition_membership(
        frame, dataset_id="fixture", train_end="2025-10", holdout_end="2025-11"
    )
    params = {"n_estimators": 3, "max_depth": 10, "min_samples_leaf": 2, "max_features": 0.5}
    fitted = fit_final_variants(
        frame,
        membership,
        selected_family="Random Forest",
        params=params,
        full_features=MODEL_FEATURES,
        top2_features=["job_category", "years_of_experience"],
        seed=42,
    )
    assert set(fitted.pipelines) == {"final_full", "final_top2"}
    assert fitted.fit_rows == int((membership["partition"] == "TRAIN").sum())
    assert fitted.params == params
    assert fitted.train_row_ids == tuple(membership.loc[membership["partition"] == "TRAIN", "row_id"])


def test_full_and_top2_variants_use_same_fold_parameters_and_membership() -> None:
    calls = []

    def evaluate(variant, features, params, fold_id):
        calls.append((variant, tuple(features), params.copy(), fold_id))
        return {"validation_MAE": 10.0 if variant == "full" else 12.0}

    winners = {
        "outer-1": {"n_estimators": 50, "max_depth": 10, "min_samples_leaf": 2, "max_features": 0.5},
        "outer-2": {"n_estimators": 100, "max_depth": 15, "min_samples_leaf": 4, "max_features": 0.6},
    }
    rows = evaluate_matched_variants(
        fold_winners=winners,
        full_features=["job_category", "years_of_experience", "city"],
        top2_features=["job_category", "years_of_experience"],
        evaluate=evaluate,
    )
    table = pd.DataFrame(rows)
    assert len(table) == 4
    for fold_id, group in table.groupby("fold_id"):
        assert set(group["variant"]) == {"full", "top2"}
        fold_calls = [call for call in calls if call[3] == fold_id]
        assert fold_calls[0][2] == fold_calls[1][2] == winners[fold_id]
