from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge

from ai_job_market.core import MODEL_FEATURES, TARGET
from ai_job_market.training_evaluation import (
    derive_performance_comparison,
    evaluate_frozen_models,
    extract_rf_fold_importance,
    fit_diagnostics,
    permutation_mae_importance,
    regression_metrics_strict,
    run_feature_family_ablation,
    score_frozen_pipelines,
    select_family,
    subgroup_metrics,
    summarize_candidates,
    uncertainty_evidence,
)
from ai_job_market.training_partitions import (
    build_expanding_monthly_folds,
    build_partition_membership,
)


def salary_frame() -> pd.DataFrame:
    rows = []
    for month in range(1, 13):
        for item in range(4):
            years = item + month / 10
            rows.append(
                {
                    "posting_year": 2025,
                    "posting_month": month,
                    "job_title": f"Role-{month}" if month >= 8 else "Role-early",
                    "job_category": "Data Science" if item % 2 else "AI Engineering",
                    "years_of_experience": years,
                    "education_required": "Bachelor's",
                    "city": "Hanoi",
                    "country": "Vietnam",
                    "remote_work": "Hybrid",
                    "company_size": "Mid-size (501-5000)",
                    "industry": "Technology",
                    "demand_score": 50 + month,
                    "benefits_score_10": 6 + item / 10,
                    "required_skills": "Python|SQL",
                    "skill_count": 2,
                    TARGET: 50_000 + 3_000 * years + 250 * month,
                }
            )
    return pd.DataFrame(rows)


def small_models() -> dict:
    return {
        "Dummy Median": DummyRegressor(strategy="median"),
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=10.0),
        "Random Forest": RandomForestRegressor(n_estimators=5, random_state=42, n_jobs=1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=5, random_state=42),
    }


def training_inputs():
    frame = salary_frame()
    membership = build_partition_membership(
        frame, dataset_id="fixture", train_end="2025-10", holdout_end="2025-11"
    )
    train_membership = membership[membership["partition"] == "TRAIN"]
    folds = build_expanding_monthly_folds(
        train_membership,
        n_splits=5,
        scope="outer",
        parent_population_id="TRAIN:fixture",
    )
    return frame, membership, folds


def test_five_models_use_identical_folds_and_only_training_rows() -> None:
    frame, membership, folds = training_inputs()
    result = evaluate_frozen_models(frame, membership, MODEL_FEATURES, small_models(), folds)

    assert len(result.fold_metrics) == 25
    expected = {fold.fold_id: (len(fold.train_row_ids), len(fold.validation_row_ids)) for fold in folds}
    for fold_id, rows in result.fold_metrics.groupby("fold_id"):
        assert len(rows) == 5
        assert set(zip(rows["train_rows"], rows["validation_rows"], strict=True)) == {
            expected[fold_id]
        }
    protected = set(
        membership.loc[membership["partition"] != "TRAIN", "row_id"]
    )
    assert not (set(result.oof_predictions["row_id"]) & protected)
    assert result.oof_predictions.groupby("model")["row_id"].nunique().eq(20).all()


def test_fold_local_preprocessing_does_not_learn_future_categories() -> None:
    frame, membership, folds = training_inputs()
    result = evaluate_frozen_models(
        frame, membership, MODEL_FEATURES, {"Dummy Median": DummyRegressor(strategy="median")}, folds
    )
    counts = result.fold_metrics.sort_values("fold_ordinal")["encoded_feature_count"].tolist()
    assert counts == sorted(counts)
    assert counts[0] < counts[-1]


def test_strict_metrics_use_null_reason_for_undefined_r2() -> None:
    metrics = regression_metrics_strict([10.0], [9.0])
    assert metrics["R2"] is None
    assert metrics["R2_reason"] == "fewer_than_two_rows"

    metrics = regression_metrics_strict([10.0, 10.0], [9.0, 11.0])
    assert metrics["R2"] is None
    assert metrics["R2_reason"] == "constant_target"
    assert metrics["MAE"] == 1.0


def test_summary_uses_equal_fold_mean_and_keeps_pooled_metrics_separate() -> None:
    frame, membership, folds = training_inputs()
    result = evaluate_frozen_models(frame, membership, MODEL_FEATURES, small_models(), folds)
    summary = summarize_candidates(result.fold_metrics, result.oof_predictions)

    linear_rows = result.fold_metrics[result.fold_metrics["model"] == "Linear Regression"]
    linear_summary = summary[summary["model"] == "Linear Regression"].iloc[0]
    assert linear_summary["cv_mae_mean_usd"] == pytest.approx(
        linear_rows["validation_MAE"].mean()
    )
    assert linear_summary["cv_mae_sd_usd"] == pytest.approx(
        linear_rows["validation_MAE"].std(ddof=0)
    )
    assert linear_summary["valid_fold_count"] == 5
    assert linear_summary["pooled_oof_mae_usd"] >= 0


def test_selection_uses_overlap_then_declared_simplicity_not_runtime() -> None:
    summary = pd.DataFrame(
        [
            {"model": "Random Forest", "cv_mae_mean_usd": 100.0, "cv_mae_sd_usd": 8.0},
            {"model": "Linear Regression", "cv_mae_mean_usd": 105.0, "cv_mae_sd_usd": 5.0},
            {"model": "Dummy Median", "cv_mae_mean_usd": 150.0, "cv_mae_sd_usd": 5.0},
            {"model": "Ridge Regression", "cv_mae_mean_usd": 130.0, "cv_mae_sd_usd": 5.0},
            {"model": "Gradient Boosting", "cv_mae_mean_usd": 140.0, "cv_mae_sd_usd": 5.0},
        ]
    )
    selection = select_family(summary)
    assert selection["lowest_mae_model"] == "Random Forest"
    assert selection["selected_family"] == "Linear Regression"
    assert selection["overlap_models"] == ["Linear Regression", "Random Forest"]


def test_derived_performance_evidence_keeps_scientific_selection_unchanged() -> None:
    predictions = pd.DataFrame(
        [
            {"model": "Dummy Median", "residual": 10.0, "absolute_error": 10.0},
            {"model": "Dummy Median", "residual": -20.0, "absolute_error": 20.0},
            {"model": "Ridge Regression", "residual": 2.0, "absolute_error": 2.0},
            {"model": "Ridge Regression", "residual": -4.0, "absolute_error": 4.0},
        ]
    )
    variants = pd.DataFrame(
        [
            {"fold_id": "outer-1", "variant": "full", "validation_MAE": 8.0},
            {"fold_id": "outer-1", "variant": "top2", "validation_MAE": 10.0},
            {"fold_id": "outer-2", "variant": "full", "validation_MAE": 12.0},
            {"fold_id": "outer-2", "variant": "top2", "validation_MAE": 11.0},
        ]
    )
    selection = {"selected_family": "Ridge Regression"}
    evidence = derive_performance_comparison(
        predictions,
        variant_fold_metrics=variants,
        tuning_baseline_mae=9.0,
        tuning_selected_mae=8.0,
    )
    ridge = evidence["models"].query("model == 'Ridge Regression'").iloc[0]
    assert ridge["mean_residual_usd"] == -1.0
    assert ridge["dummy_skill_mae"] == 0.8
    assert evidence["top2_mae_delta_usd"] == 0.5
    assert evidence["tuning_mae_delta_usd"] == -1.0
    assert selection == {"selected_family": "Ridge Regression"}


def test_fit_diagnostics_are_paired_to_dummy_and_do_not_force_mixed_labels() -> None:
    frame, membership, folds = training_inputs()
    result = evaluate_frozen_models(frame, membership, MODEL_FEATURES, small_models(), folds)
    diagnostics = fit_diagnostics(result.fold_metrics, gap_threshold=0.2)
    assert set(diagnostics["model"]) == set(small_models())
    assert diagnostics[diagnostics["model"] == "Dummy Median"]["fold_indication"].eq(
        "baseline"
    ).all()
    assert diagnostics["aggregate_indication"].notna().all()
    assert diagnostics["normalized_mae_gap"].dropna().map(np.isfinite).all()


def test_feature_family_ablation_uses_only_approved_subsets() -> None:
    frame, membership, folds = training_inputs()
    families = {
        "job_domain": ["job_title", "job_category"],
        "experience_education": ["years_of_experience", "education_required"],
        "geography": ["city", "country"],
        "company_work": ["remote_work", "company_size", "industry"],
        "demand_benefits": ["demand_score", "benefits_score_10"],
        "skills": ["required_skills", "skill_count"],
    }
    model = RandomForestRegressor(n_estimators=3, random_state=42, n_jobs=1)
    baseline = evaluate_frozen_models(
        frame, membership, MODEL_FEATURES, {"Random Forest": model}, folds
    )
    summary, fold_rows = run_feature_family_ablation(
        frame,
        membership,
        folds,
        features=MODEL_FEATURES,
        families=families,
        model=model,
        baseline_fold_metrics=baseline.fold_metrics,
    )
    assert len(summary) == 6
    assert len(fold_rows) == 30
    assert "experience_level" not in "|".join(summary["features"])
    assert set(summary["removed_family"]) == set(families)

    prohibited = families | {"bad": ["experience_level"]}
    with pytest.raises(ValueError, match="approved feature policy"):
        run_feature_family_ablation(
            frame,
            membership,
            folds,
            features=MODEL_FEATURES,
            families=prohibited,
            model=RandomForestRegressor(n_estimators=2, random_state=42, n_jobs=1),
            baseline_fold_metrics=baseline.fold_metrics,
        )


def test_rf_importance_is_captured_from_existing_fits_and_drift_is_bounded() -> None:
    frame, membership, folds = training_inputs()
    result = evaluate_frozen_models(
        frame,
        membership,
        MODEL_FEATURES,
        {"Random Forest": RandomForestRegressor(n_estimators=3, random_state=42, n_jobs=1)},
        folds,
    )
    pipeline_ids = {key: id(value) for key, value in result.fitted_pipelines.items()}
    families = {
        "job_domain": ["job_title", "job_category"],
        "experience_education": ["years_of_experience", "education_required"],
        "geography": ["city", "country"],
        "company_work": ["remote_work", "company_size", "industry"],
        "demand_benefits": ["demand_score", "benefits_score_10"],
        "skills": ["required_skills", "skill_count"],
    }
    encoded, family, drift = extract_rf_fold_importance(
        result, features=MODEL_FEATURES, families=families
    )
    assert len(encoded) > 0
    assert set(family["family"]) == set(families)
    assert len(drift) == 4
    assert drift["half_l1_drift"].between(0, 1).all()
    assert pipeline_ids == {key: id(value) for key, value in result.fitted_pipelines.items()}


def test_final_scoring_uses_frozen_pipelines_and_holdout_only(monkeypatch) -> None:
    frame, membership, _ = training_inputs()
    train_positions = membership.loc[membership["partition"] == "TRAIN", "source_position"]
    from ai_job_market.core import make_model_pipeline

    pipelines = {}
    features_by_model = {}
    for role in [
        "Dummy Median",
        "Linear Regression",
        "Ridge Regression",
        "Random Forest",
        "Gradient Boosting",
        "final_full",
        "final_top2",
    ]:
        features = (
            ["job_category", "years_of_experience"] if role == "final_top2" else MODEL_FEATURES
        )
        pipeline = make_model_pipeline(DummyRegressor(strategy="median"), features)
        pipeline.fit(frame.iloc[train_positions][features], frame.iloc[train_positions][TARGET])
        monkeypatch.setattr(
            pipeline,
            "fit",
            lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not refit")),
        )
        pipelines[role] = pipeline
        features_by_model[role] = features

    metrics, predictions = score_frozen_pipelines(
        frame, membership, pipelines=pipelines, features_by_model=features_by_model
    )
    assert len(metrics) == 7
    assert set(predictions["model_role_id"]) == set(pipelines)
    holdout_ids = set(membership.loc[membership["partition"] == "EVALUATION_HOLDOUT", "row_id"])
    reserve_ids = set(membership.loc[membership["partition"] == "INFERENCE_RESERVE", "row_id"])
    assert set(predictions["row_id"]) == holdout_ids
    assert not set(predictions["row_id"]) & reserve_ids
    assert np.allclose(predictions["residual"], predictions["actual"] - predictions["predicted"])


def test_permutation_subgroups_and_q90_are_reproducible_and_scoped() -> None:
    frame, membership, _ = training_inputs()
    train_positions = membership.loc[membership["partition"] == "TRAIN", "source_position"]
    holdout_positions = membership.loc[
        membership["partition"] == "EVALUATION_HOLDOUT", "source_position"
    ]
    from ai_job_market.core import make_model_pipeline

    pipeline = make_model_pipeline(
        RandomForestRegressor(n_estimators=5, random_state=42, n_jobs=1), MODEL_FEATURES
    )
    pipeline.fit(frame.iloc[train_positions][MODEL_FEATURES], frame.iloc[train_positions][TARGET])
    families = {
        "job_domain": ["job_title", "job_category"],
        "experience_education": ["years_of_experience", "education_required"],
        "geography": ["city", "country"],
        "company_work": ["remote_work", "company_size", "industry"],
        "demand_benefits": ["demand_score", "benefits_score_10"],
        "skills": ["required_skills", "skill_count"],
    }
    importance = permutation_mae_importance(
        pipeline,
        frame.iloc[holdout_positions],
        features=MODEL_FEATURES,
        families=families,
        repeats=12,
        seed=42,
    )
    assert set(importance["kind"]) == {"raw_feature", "joint_family"}
    assert importance.groupby(["kind", "name"]).size().eq(12).all()
    assert np.isfinite(importance["mae_increase"]).all()

    _, predictions = score_frozen_pipelines(
        frame,
        membership,
        pipelines={"final_full": pipeline},
        features_by_model={"final_full": MODEL_FEATURES},
    )
    slices = subgroup_metrics(predictions)
    assert {"job_category", "country", "experience_bucket"} <= set(slices["subgroup"])
    uncertainty, with_bands = uncertainty_evidence(predictions)
    expected = float(np.quantile(predictions["absolute_error"], 0.9, method="linear"))
    assert uncertainty["q90_absolute_error_usd"] == pytest.approx(expected)
    assert np.allclose(with_bands["lower"], with_bands["predicted"] - expected)
    assert uncertainty["basis"] == "same_holdout_descriptive_not_calibrated"


def test_invalid_model_prediction_fails_instead_of_dropping_fold() -> None:
    frame, membership, folds = training_inputs()

    class NonfiniteDummy(DummyRegressor):
        def predict(self, X):
            return np.full(len(X), np.nan)

    with pytest.raises(ValueError, match="non-finite"):
        evaluate_frozen_models(
            frame,
            membership,
            MODEL_FEATURES,
            {"Broken": NonfiniteDummy(strategy="median")},
            folds,
        )
