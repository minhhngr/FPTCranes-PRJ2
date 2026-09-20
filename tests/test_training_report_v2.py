from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ai_job_market.training_report import (
    build_agent_summary,
    build_fold_explanation,
    build_metric_catalog,
    build_model_conclusions,
    build_ui_summary,
    evaluate_outcome,
    render_report,
    write_report_charts,
)


def fold_summary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "fold_id": f"outer-{index}",
                "scope": "outer",
                "ordinal": index,
                "train_period_min": "2025-01",
                "train_period_max": f"2025-{index + 4:02d}",
                "validation_period_min": f"2025-{index + 5:02d}",
                "train_months": [f"2025-{month:02d}" for month in range(1, index + 5)],
                "validation_months": [f"2025-{index + 5:02d}"],
                "train_rows": 10 * (index + 4),
                "validation_rows": 10 + index,
                "not_used_yet_rows": sum(10 + later for later in range(index + 1, 6)),
                "added_train_rows": None if index == 1 else 9 + index,
                "added_train_months": None if index == 1 else [f"2025-{index + 4:02d}"],
                "row_overlap_count": 0,
                "shared_month_count": 0,
                "chronological_order_ok": True,
                "expanding_history_ok": None if index == 1 else True,
                "holdout_rows_excluded": 19,
                "reserve_rows_excluded": 1,
                "evidence_ref": f"fold_summary.csv#outer-{index}",
            }
            for index in range(1, 6)
        ]
    )


def candidate_summary() -> pd.DataFrame:
    names = [
        "Dummy Median",
        "Linear Regression",
        "Ridge Regression",
        "Random Forest",
        "Gradient Boosting",
    ]
    return pd.DataFrame(
        [
            {
                "model": name,
                "configuration_id": f"config-{index}",
                "valid_fold_count": 5,
                "failed_fold_count": 0,
                "cv_mae_mean_usd": 20_000 - 1_000 * index,
                "cv_mae_sd_usd": 1_000 + index,
                "cv_rmse_mean_usd": 25_000 - 900 * index,
                "cv_medae_mean_usd": 18_000 - 800 * index,
                "cv_r2_mean": 0.1 * index,
                "worst_fold_id": "outer-3",
                "total_cv_fit_wall_s": 0.2 + index,
            }
            for index, name in enumerate(names)
        ]
    )


def test_fold_explanation_shows_every_outer_fold_and_exact_rows() -> None:
    text = build_fold_explanation(
        fold_summary(),
        partition_counts={"TRAIN": 80, "EVALUATION_HOLDOUT": 19, "INFERENCE_RESERVE": 1},
    )
    assert "CV splits only the TRAIN partition" in text
    assert "Outer fold 3/5" in text
    assert "Train: 70 rows" in text
    assert "Validate: 13 rows" in text
    assert "19 rows" in text and "1 rows" in text
    assert "expanding" in text.lower()


def test_every_candidate_and_final_variant_has_a_scoped_conclusion() -> None:
    selection = {
        "lowest_mae_model": "Gradient Boosting",
        "selected_family": "Random Forest",
        "overlap_models": ["Random Forest", "Gradient Boosting"],
    }
    conclusions = build_model_conclusions(
        candidate_summary(),
        selection=selection,
        final_metrics=pd.DataFrame(
            [
                {"model_role_id": "final_full", "MAE": 14_000.0, "RMSE": 18_000.0, "R2": 0.5, "MedAE": 13_000.0},
                {"model_role_id": "final_top2", "MAE": 16_000.0, "RMSE": 20_000.0, "R2": 0.4, "MedAE": 15_000.0},
            ]
        ),
    )
    assert len(conclusions) == 7
    assert {record["model_role_id"] for record in conclusions} == {
        "Dummy Median",
        "Linear Regression",
        "Ridge Regression",
        "Random Forest",
        "Gradient Boosting",
        "final_full",
        "final_top2",
    }
    by_id = {record["model_role_id"]: record for record in conclusions}
    assert by_id["Gradient Boosting"]["decision"] == "lowest_cv_mae_not_selected"
    assert by_id["Random Forest"]["decision"] == "selected_family"
    assert by_id["Dummy Median"]["fit_diagnosis"] == "baseline"
    assert by_id["final_top2"]["decision"] == "comparison_only_no_automatic_promotion"
    assert all(record["evidence_refs"] for record in conclusions)


def test_missing_candidate_evidence_stays_visible_and_is_not_favorable() -> None:
    summary = candidate_summary().query("model != 'Ridge Regression'")
    conclusions = build_model_conclusions(
        summary,
        selection={"lowest_mae_model": "Gradient Boosting", "selected_family": "Gradient Boosting", "overlap_models": ["Gradient Boosting"]},
        final_metrics=pd.DataFrame(),
    )
    ridge = next(item for item in conclusions if item["model_role_id"] == "Ridge Regression")
    assert ridge["status"] == "unavailable"
    assert ridge["decision"] == "not_eligible_missing_evidence"
    assert "missing" in ridge["limitation"].lower()


def test_human_agent_and_ui_views_reference_the_same_values() -> None:
    summary = candidate_summary()
    selection = {
        "lowest_mae_model": "Gradient Boosting",
        "selected_family": "Random Forest",
        "overlap_models": ["Random Forest", "Gradient Boosting"],
    }
    conclusions = build_model_conclusions(summary, selection=selection, final_metrics=pd.DataFrame())
    folds = fold_summary()
    agent = build_agent_summary(
        run_id="tv-" + "a" * 32,
        summary=summary,
        selection=selection,
        fold_summary=folds,
        conclusions=conclusions,
    )
    ui = build_ui_summary(
        run_id="tv-" + "a" * 32,
        manifest_sha256="b" * 64,
        summary=summary,
        selection=selection,
        fold_summary=folds,
        conclusions=conclusions,
    )
    assert agent["selected_family_id"] == ui["page04"]["conclusion"]["selected_family_id"]
    assert agent["lowest_cv_mae_model_id"] == ui["page04"]["conclusion"]["lowest_cv_mae_model_id"]
    assert agent["folds"][2]["train_rows"] == ui["page04"]["fold_method"]["rows"][2]["train_rows"]
    assert len(ui["page04"]["model_conclusions"]) == 5
    json.dumps(agent, allow_nan=False)
    json.dumps(ui, allow_nan=False)


def test_outcome_rules_do_not_invent_missing_thresholds_or_evidence() -> None:
    assert evaluate_outcome(None, {"max_holdout_mae_usd": 10})["status"] == "Blocked"
    assert evaluate_outcome({"MAE": 8, "RMSE": 9}, {})["status"] == "Inconclusive"
    assert evaluate_outcome(
        {"MAE": 8, "RMSE": 9},
        {"max_holdout_mae_usd": 10, "max_holdout_rmse_usd": 10},
    )["status"] == "Good"
    assert evaluate_outcome(
        {"MAE": 11, "RMSE": 9},
        {"max_holdout_mae_usd": 10, "max_holdout_rmse_usd": 10},
    )["status"] == "Bad"


def test_report_answers_5w1h_and_classification_is_not_applicable() -> None:
    report = render_report(
        run_context={
            "who": "local operator and data custodian",
            "what": "continuous annual salary regression",
            "when": "2025-01 through approved periods",
            "where": "contained offline training workspace",
            "why": "compare models without future leakage",
            "how": "whole-month partitions and expanding monthly folds",
        },
        fold_explanation=build_fold_explanation(
            fold_summary(),
            partition_counts={"TRAIN": 80, "EVALUATION_HOLDOUT": 19, "INFERENCE_RESERVE": 1},
        ),
        model_conclusions=build_model_conclusions(
            candidate_summary(),
            selection={"lowest_mae_model": "Gradient Boosting", "selected_family": "Random Forest", "overlap_models": ["Random Forest", "Gradient Boosting"]},
            final_metrics=pd.DataFrame(),
        ),
    )
    for question in ("Who", "What", "When", "Where", "Why", "How"):
        assert f"**{question}**" in report
    assert "Classification metrics: not applicable" in report
    assert "Finding" in report and "Limitation" in report and "Next action" in report


def test_metric_catalog_and_charts_are_generated_from_authoritative_tables(tmp_path: Path) -> None:
    catalog = build_metric_catalog()
    assert catalog["mae_usd"]["unit"] == "USD/year"
    assert catalog["r2"]["direction"] == "higher"
    paths = write_report_charts(
        tmp_path,
        candidate_summary(),
        holdout_predictions=pd.DataFrame(
            {"actual": [10.0, 20.0], "predicted": [11.0, 18.0], "residual": [-1.0, 2.0]}
        ),
        importance=pd.DataFrame({"name": ["a", "b"], "mae_increase": [3.0, 1.0]}),
    )
    assert {path.name for path in paths} == {
        "model_comparison.html",
        "accuracy_runtime_tradeoff.html",
        "holdout_diagnostics.html",
        "feature_importance.html",
    }
    assert all(path.stat().st_size > 0 for path in paths)
