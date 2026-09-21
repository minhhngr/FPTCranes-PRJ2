from __future__ import annotations

import pandas as pd

from ai_job_market.core import MODEL_FEATURES, TARGET
from ai_job_market.training_partitions import (
    build_expanding_monthly_folds,
    build_partition_membership,
)
from ai_job_market.training_search import (
    conditional_gridsearch_rf_search,
    evaluate_matched_variants,
    fit_final_variants,
    gridsearch_rf_search,
)


def search_policy() -> dict:
    return {
        "n_estimators": [3, 5],
        "max_depth": [2, None],
        "min_samples_leaf": 1,
        "max_features": 1.0,
    }


def training_context() -> tuple[pd.DataFrame, pd.DataFrame, list]:
    rows = []
    for month in range(1, 13):
        for item in range(3):
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
                    TARGET: 60_000 + month * 1_000 + item * 3_000,
                }
            )
    frame = pd.DataFrame(rows)
    membership = build_partition_membership(
        frame, dataset_id="fixture", train_end="2025-10", holdout_end="2025-11"
    )
    train = membership[membership["partition"] == "TRAIN"]
    folds = build_expanding_monthly_folds(
        train, n_splits=3, scope="inner", parent_population_id="TRAIN"
    )
    return frame, membership, folds


def test_gridsearch_records_cartesian_grid_temporal_folds_and_r2() -> None:
    frame, membership, folds = training_context()
    result = gridsearch_rf_search(
        search_id="rf-final-train",
        frame=frame,
        membership=membership,
        features=MODEL_FEATURES,
        folds=folds,
        policy=search_policy(),
        seed=42,
    )

    assert len(result.trials) == 4
    assert result.trials["fold_count"].eq(3).all()
    assert result.trials["ranking_metric"].eq("MAE").all()
    assert result.trials["scoring"].eq("neg_mean_absolute_error").all()
    assert result.trials["inner_fold_ids"].map(len).eq(3).all()
    assert result.trials["n_estimators"].isin([3, 5]).all()
    assert result.trials["max_depth"].map(lambda value: value == 2 or pd.isna(value)).all()
    assert result.trials["mae_mean"].notna().all()
    assert result.trials["r2_mean"].notna().all()
    assert result.trials["method_version"].eq("gridsearchcv-temporal/v1").all()
    assert result.trials["evidence_ref"].str.startswith("tuning_trials.csv#trial_id=").all()
    assert result.trials["fold_r2"].map(len).eq(3).all()
    assert result.trials["fold_mae"].map(len).eq(3).all()
    assert result.winner is not None
    assert result.actual_fit_count == 12


def test_conditional_gridsearch_skips_when_random_forest_is_not_selected() -> None:
    frame, membership, folds = training_context()
    result = conditional_gridsearch_rf_search(
        selected_family="Linear Regression",
        search_id="skip",
        frame=frame,
        membership=membership,
        features=MODEL_FEATURES,
        folds=folds,
        policy=search_policy(),
        seed=42,
    )
    assert result.status == "skipped"
    assert result.reason == "selected_family_is_not_random_forest"
    assert result.trials.empty


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
    assert fitted.train_row_ids == tuple(
        membership.loc[membership["partition"] == "TRAIN", "row_id"]
    )


def test_full_and_top2_variants_use_same_fold_parameters_and_membership() -> None:
    calls = []

    def evaluate(variant, features, params, fold_id):
        calls.append((variant, tuple(features), params.copy(), fold_id))
        return {"validation_MAE": 10.0 if variant == "full" else 12.0}

    winners = {
        "outer-1": {
            "n_estimators": 50,
            "max_depth": 10,
            "min_samples_leaf": 2,
            "max_features": 0.5,
        },
        "outer-2": {
            "n_estimators": 100,
            "max_depth": 15,
            "min_samples_leaf": 4,
            "max_features": 0.6,
        },
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
