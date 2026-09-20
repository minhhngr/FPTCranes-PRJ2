from __future__ import annotations

import pandas as pd

from pages.model_evidence import (
    CHART_WIDTHS,
    EVIDENCE_COLORS,
    actual_predicted_figure,
    baseline_status,
    candidate_comparison_figure,
    chart_container_width,
    fold_stability_conclusion,
    generalization_conclusion,
    generalization_error_figure,
    importance_figure,
    prediction_batch_conclusion,
    prediction_interval_figure,
    ranking_conclusion,
    uncertainty_conclusion,
    variant_conclusion,
)
from pages.model_training_presentation import (
    candidate_decision_table,
    temporal_validation_guide,
    tuning_decision_table,
    tuning_method_guide,
)


def test_candidate_decision_table_uses_unrounded_ties_and_markdown_emphasis():
    summary = pd.DataFrame(
        {
            "model": ["Winner A", "Winner B", "Dummy Median", "Rounded runner"],
            "validation_MAE_mean": [10.0001, 10.0001, 50.0, 10.0004],
            "train_MAE_mean": [8.0, 8.5, 48.0, 8.2],
            "validation_R2_mean": [0.81234, 0.81231, -0.1, 0.81235],
        }
    )

    table = candidate_decision_table(summary)

    assert list(table["Rank"]) == [1, 1, 3, 4]
    assert table.loc[0, "Model"] == "**Winner A**"
    assert table.loc[1, "Model"] == "**Winner B**"
    assert table.loc[2, "Model"] == "Rounded runner"
    assert table.loc[3, "Model"] == "*Dummy Median*"
    assert table.loc[0, "Mean validation MAE"] == "**$10**"
    assert table.loc[3, "Mean validation MAE"] == "*$50*"
    assert table.loc[0, "Validation R²"] == "**0.812**"


def test_tuning_decision_table_distinguishes_stage_winner_from_applied_value():
    summary = pd.DataFrame(
        {
            "hyperparameter": ["1. n_estimators", "2. max_depth"],
            "search_space": ["[50, 100, 200]", "[10, 20, None]"],
            "optimal_value": ["200", "20"],
            "best_cv_r2": [0.8238161, 0.8238161],
            "best_cv_mae": [15593.99, 15593.99],
        }
    )

    table = tuning_decision_table(
        summary,
        {"n_estimators": 200, "max_depth": 15},
    )

    assert table.loc[0, "Applied value"] == "**200**"
    assert table.loc[0, "Stage winner"] == "**200**"
    assert table.loc[0, "Relationship"] == "**Applied = stage winner**"
    assert table.loc[1, "Applied value"] == "**15**"
    assert table.loc[1, "Stage winner"] == "**20**"
    assert table.loc[1, "Relationship"] == "*Sensitivity only; not applied*"
    assert table.loc[0, "Search space"] == "*[50, 100, 200]*"
    assert table.loc[0, "Best CV R²"] == "**0.824**"
    assert table.loc[0, "Best CV MAE"] == "**$15,594**"


def test_temporal_validation_guide_describes_adjacent_disjoint_blocks():
    folds = pd.DataFrame(
        {
            "model": ["A", "A", "B", "B"],
            "fold_id": [1, 2, 1, 2],
            "validation_period": ["2025-02", "2025-03", "2025-02", "2025-03"],
            "train_rows": [2, 2, 2, 2],
            "validation_rows": [2, 3, 2, 3],
            "encoded_feature_count": [10, 11, 10, 11],
        }
    )
    membership = pd.DataFrame(
        {
            "fold_id": [1, 1, 1, 1, 2, 2, 2, 2, 2],
            "partition_role": [
                "train",
                "train",
                "validation",
                "validation",
                "train",
                "train",
                "validation",
                "validation",
                "validation",
            ],
            "record_id": ["a", "b", "c", "d", "c", "d", "e", "f", "g"],
        }
    )

    guide = temporal_validation_guide(folds, membership, full_feature_count=13)

    assert guide["fold_count"] == 2
    assert guide["candidate_count"] == 2
    assert guide["full_feature_count"] == 13
    assert guide["row_overlap_count"] == 0
    assert guide["folds"][1]["validation_rows"] == 3
    assert "adjacent" in guide["method"].lower()
    assert "not randomized" in guide["limitations"].lower()
    assert "not an expanding" in guide["limitations"].lower()


def test_tuning_method_guide_uses_actual_stage_rows_and_excludes_rationale():
    tables = {
        "manual_tuning_step1_gridsearch": pd.DataFrame(
            {
                "candidate": [1, 2, 3, 4],
                "CV_R2": [0.8, 0.79, 0.78, 0.82],
                "n_estimators": [300, 100, 80, 200],
                "max_depth": [20, None, None, 20],
                "min_samples_leaf": [2, 2, 1, 1],
                "max_features": [0.7, 0.8, 0.8, 0.7],
            }
        ),
        "manual_tuning_step2_n_estimators": pd.DataFrame(
            {"n_estimators": [50, 100, 200], "CV_R2": [0.80, 0.81, 0.82]}
        ),
        "manual_tuning_step3_max_depth": pd.DataFrame(
            {"max_depth": [10, 20], "CV_R2": [0.80, 0.82]}
        ),
        "manual_tuning_step4_min_samples_leaf": pd.DataFrame(
            {"min_samples_leaf": [1, 2], "CV_R2": [0.82, 0.81]}
        ),
        "manual_tuning_step5_max_features": pd.DataFrame(
            {"max_features": [0.5, 0.7], "CV_R2": [0.80, 0.82]}
        ),
        "manual_tuning_4params_summary": pd.DataFrame(
            {
                "hyperparameter": ["1. n_estimators"],
                "tuning_rationale": ["unsupported causal claim"],
            }
        ),
    }

    guide = tuning_method_guide(
        tables,
        applied_parameters={
            "n_estimators": 200,
            "max_depth": 20,
            "min_samples_leaf": 1,
            "max_features": 0.7,
        },
        effective_folds=5,
    )

    assert guide["initial_trial_count"] == 4
    assert guide["total_trial_count"] == 13
    assert guide["effective_folds"] == 5
    assert guide["ranking_metric"] == "CV R²"
    assert guide["ranking_direction"] == "higher is better"
    assert guide["saved_matches_initial_winner"] is True
    assert guide["stage_trial_counts"] == {
        "n_estimators": 3,
        "max_depth": 2,
        "min_samples_leaf": 2,
        "max_features": 2,
    }
    assert "tuning_rationale" not in str(guide)


def test_candidate_chart_uses_artifact_values_and_dummy_reference():
    frame = pd.DataFrame(
        {
            "model": ["Random Forest", "Dummy Median"],
            "validation_MAE_mean": [12.5, 55.0],
            "train_MAE_mean": [7.0, 53.0],
        }
    )
    figure = candidate_comparison_figure(frame)
    assert list(figure.data[0].y) == [12.5, 55.0]
    assert list(figure.data[1].y) == [7.0, 53.0]
    assert figure.layout.shapes[0].y0 == 55.0
    assert figure.data[0].marker.color == EVIDENCE_COLORS["validation"]
    assert figure.data[1].line.color == EVIDENCE_COLORS["training"]
    assert figure.data[1].marker.color == EVIDENCE_COLORS["training"]
    assert figure.layout.shapes[0].line.color == EVIDENCE_COLORS["threshold"]
    assert baseline_status(frame.iloc[0], 55.0) == "Below baseline"
    assert baseline_status(frame.iloc[1], 55.0) == "Reference baseline"


def test_candidate_colors_are_role_based_when_input_order_changes():
    frame = pd.DataFrame(
        {
            "model": ["Dummy Median", "Random Forest"],
            "validation_MAE_mean": [55.0, 12.5],
            "train_MAE_mean": [53.0, 7.0],
        }
    )
    figure = candidate_comparison_figure(frame)
    traces = {trace.name: trace for trace in figure.data}
    assert traces["Validation MAE"].marker.color == EVIDENCE_COLORS["validation"]
    assert traces["Train MAE"].line.color == EVIDENCE_COLORS["training"]


def test_generalization_uses_red_test_columns_and_orange_dev_line():
    cv = pd.Series({"MAE": 15.0, "MedAE": 7.0, "RMSE": 26.0})
    test = pd.Series({"MAE": 14.0, "MedAE": 4.0, "RMSE": 29.0})
    figure = generalization_error_figure(cv, test)
    traces = {trace.name: trace for trace in figure.data}
    assert traces["Historical test"].type == "bar"
    assert traces["Historical test"].marker.color == EVIDENCE_COLORS["historical_test"]
    assert traces["DEV CV"].type == "scatter"
    assert traces["DEV CV"].line.color == EVIDENCE_COLORS["development"]
    assert "lines+markers+text" == traces["DEV CV"].mode


def test_prediction_figure_keeps_actual_marker_outside_interval():
    frame = pd.DataFrame(
        {
            "scenario_id": ["s1"],
            "predicted_salary_usd": [100.0],
            "lower_bound_usd": [90.0],
            "upper_bound_usd": [110.0],
            "actual_salary_usd": [200.0],
        }
    )
    figure = prediction_interval_figure(frame)
    assert figure.data[0].marker.color == EVIDENCE_COLORS["prediction"]
    assert figure.data[0].error_y.color == EVIDENCE_COLORS["uncertainty"]
    assert figure.data[1].y[0] == 200.0
    assert figure.data[1].marker.color == EVIDENCE_COLORS["actual"]
    assert figure.data[1].marker.symbol == "diamond"


def test_chart_priority_limits_full_width_to_primary_charts():
    assert chart_container_width("P1") == "stretch"
    assert chart_container_width("P2") == CHART_WIDTHS["P2"]
    assert chart_container_width("P3") == CHART_WIDTHS["P3"]


def test_candidate_chart_reserves_space_and_reduces_comparison_line_labels():
    frame = pd.DataFrame(
        {
            "model": ["Long Linear Regression", "Ridge Regression", "Random Forest"],
            "validation_MAE_mean": [59_000.0, 28_000.0, 15_000.0],
            "train_MAE_mean": [11_000.0, 18_000.0, 7_000.0],
        }
    )
    figure = candidate_comparison_figure(frame)
    layout = figure.layout
    assert layout.height >= 460
    assert layout.margin.b >= 80
    assert layout.xaxis.automargin is True
    assert layout.yaxis.tickformat == ",.0f"
    assert sum(bool(label) for label in figure.data[1].text) < len(frame)


def test_dense_actual_predicted_scatter_uses_hover_not_point_text_labels():
    frame = pd.DataFrame(
        {
            "model_id": ["m"] * 25,
            "annual_salary_usd": range(25),
            "predicted_salary_usd": range(1, 26),
            "job_title": [f"Role {index}" for index in range(25)],
        }
    )
    figure = actual_predicted_figure(frame, "m")
    assert figure.data[0].mode == "markers"
    assert "text" not in figure.data[0].mode


def test_horizontal_importance_reserves_feature_label_margin_and_height():
    frame = pd.DataFrame(
        {
            "model_id": ["m"] * 13,
            "raw_feature": [f"very_long_feature_name_{index}" for index in range(13)],
            "mae_increase_usd": list(range(13)),
            "std_usd": [0.1] * 13,
        }
    )
    figure = importance_figure(frame, "m")
    assert figure.layout.margin.l >= 180
    assert figure.layout.height >= 480
    assert figure.layout.yaxis.automargin is True
    assert len(figure.data[0].text) == 13


def test_ranking_and_fold_conclusions_are_derived_from_changed_evidence():
    summary = pd.DataFrame(
        {
            "model": ["Challenger", "Winner", "Dummy Median"],
            "validation_MAE_mean": [20.0, 10.0, 50.0],
        }
    )
    conclusion = ranking_conclusion(summary)
    assert "Winner" in conclusion["finding"]
    assert "$10" in conclusion["finding"]
    assert "$10" in conclusion["why_it_matters"]
    assert "80.0%" in conclusion["why_it_matters"]

    folds = pd.DataFrame(
        {
            "model": ["Winner"] * 3,
            "fold_id": [1, 2, 3],
            "validation_MAE": [9.0, 25.0, 11.0],
            "validation_period": ["A", "B", "C"],
        }
    )
    stability = fold_stability_conclusion(folds, "Winner")
    assert "Fold 2" in stability["finding"]
    assert "B" in stability["finding"]


def test_generalization_uncertainty_and_variant_conclusions_use_correct_directions():
    cv = pd.Series({"MAE": 20.0, "MedAE": 10.0, "RMSE": 30.0, "R2": 0.80})
    test = pd.Series(
        {
            "MAE": 15.0,
            "MedAE": 8.0,
            "RMSE": 35.0,
            "R2": 0.75,
            "q90_abs_error_usd": 40.0,
            "coverage": 0.9,
            "row_count": 100,
        }
    )
    general = generalization_conclusion(cv, test)
    assert "decreased" in general["finding"]
    assert "$5" in general["finding"]
    assert "R² decreased" in general["limit"]
    uncertainty = uncertainty_conclusion(test)
    assert "$40" in uncertainty["finding"]
    assert "90.0%" in uncertainty["finding"]
    assert "100" in uncertainty["finding"]

    variants = pd.DataFrame(
        {
            "feature_variant": ["full", "top2"],
            "evaluation": ["historical_test", "historical_test"],
            "MAE": [15.0, 13.0],
            "row_count": [100, 100],
        }
    )
    variant = variant_conclusion(variants)
    assert "Top-2" in variant["finding"]
    assert "$2" in variant["finding"]


def test_prediction_conclusion_handles_actuals_exceptions_and_missing_rows():
    rows = pd.DataFrame(
        {
            "predicted_salary_usd": [100.0, 200.0],
            "absolute_error_usd": [10.0, pd.NA],
            "validation_mode": ["strict", "experience_exception"],
        }
    )
    conclusion = prediction_batch_conclusion(rows, q90_abs_error_usd=40.0)
    assert "2 scenarios" in conclusion["finding"]
    assert "$100" in conclusion["finding"] and "$200" in conclusion["finding"]
    assert "1 known actual" in conclusion["why_it_matters"]
    assert "1 extrapolated" in conclusion["limit"]

    unavailable = prediction_batch_conclusion(pd.DataFrame(), q90_abs_error_usd=40.0)
    assert unavailable["available"] is False
    assert "Conclusion unavailable" in unavailable["finding"]

    nonfinite = prediction_batch_conclusion(
        pd.DataFrame({"predicted_salary_usd": [float("inf")], "validation_mode": ["strict"]}),
        q90_abs_error_usd=40.0,
    )
    assert nonfinite["available"] is False
