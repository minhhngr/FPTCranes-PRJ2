from __future__ import annotations

import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from pages.model_evidence import EVIDENCE_COLORS

ROOT = Path(__file__).resolve().parents[1]


def open_admin_page(label: str) -> AppTest:
    app = AppTest.from_file(str(ROOT / "streamlit.py"), default_timeout=30)
    app.session_state["auth_role"] = "admin"
    app.run()
    assert not app.exception
    app.radio[0].set_value(label).run()
    return app


@pytest.mark.parametrize(
    "label,title",
    [
        ("4. Model Comparison", "4. Model comparison and temporal validation"),
        ("5. Best Model & Importance", "5. Best model, diagnostics and uncertainty"),
        ("6. Salary Prediction", "6. Controlled salary scenarios"),
    ],
)
def test_model_pages_open_from_real_entrypoint(label, title):
    app = open_admin_page(label)
    assert not app.exception
    assert any(item.value == title for item in app.title)


@pytest.mark.parametrize(
    ("label", "training_expander"),
    [
        (
            "4. Model Comparison",
            "Latest training validation: folds and candidate models",
        ),
        (
            "5. Best Model & Importance",
            "Latest training validation: tuning and Full/Top-2 variants",
        ),
    ],
)
def test_training_validation_unavailable_is_truthful_collapsed_and_actionable(
    label, training_expander
):
    app = open_admin_page(label)
    assert any(item.value == "Training & validation evidence" for item in app.subheader)
    assert any(
        "No verified training-validation evidence" in item.value
        for item in app.warning
    )
    expanders = [item for item in app.expander if item.label == training_expander]
    assert len(expanders) == 1
    assert expanders[0].proto.expanded is False
    if label.startswith("5."):
        final_expanders = [
            item
            for item in app.expander
            if item.label
            == "Latest training validation: holdout, explainability and uncertainty"
        ]
        assert len(final_expanders) == 1
        assert final_expanders[0].proto.expanded is False
    assert any(item.value == "Training validation unavailable" for item in app.info)
    assert any("training_validation.py inspect --workspace ." in item.value for item in app.code)
    rendered = "\n".join(
        str(item.value) for item in [*app.markdown, *app.caption, *app.code]
    )
    assert "tests/fixtures" not in rendered
    assert "/home/" not in rendered


@pytest.mark.parametrize(
    ("label", "expected_models", "expected_record_count"),
    [
        (
            "4. Model Comparison",
            [
                "Random Forest",
                "Gradient Boosting",
                "Ridge Regression",
                "Dummy Median",
                "Linear Regression",
            ],
            5,
        ),
        (
            "5. Best Model & Importance",
            ["Random Forest (selected family)", "Full 13", "Top 2"],
            3,
        ),
    ],
)
def test_current_pages_show_historical_5w1h_report_and_raw_training_log(
    monkeypatch, label, expected_models, expected_record_count
):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("training/model operation reached by historical report UI")

    monkeypatch.setattr("joblib.load", forbidden)
    monkeypatch.setattr("ai_job_market.training_validation.run_workspace", forbidden)
    monkeypatch.setattr("ai_job_market.training_validation.fit_final_variants", forbidden)
    monkeypatch.setattr("ai_job_market.training_validation.score_frozen_pipelines", forbidden)
    monkeypatch.setattr("ai_job_market.training_validation.publish_staged_pack", forbidden)

    app = open_admin_page(label)
    assert not app.exception
    assert any(item.value == "Historical Branch B training report" for item in app.subheader)
    assert any("Historical primary-pipeline + supplemental evidence" in item.value for item in app.warning)
    markdown = [item.value for item in app.markdown]
    assert sum(value.startswith("**Who:**") for value in markdown) == expected_record_count
    for question in ("What", "When", "Where", "Why", "How"):
        assert sum(value.startswith(f"**{question}:**") for value in markdown) == expected_record_count
    for model in expected_models:
        assert f"#### {model}" in markdown
    assert any("Branch A cluster assignments are not salary-model features" in value for value in markdown)
    assert "Historical training activity log and downloads" in [
        item.label for item in app.expander
    ]
    event_tables = [
        item.value
        for item in app.dataframe
        if {"sequence", "operation", "status", "evidence_ref"} <= set(item.value.columns)
    ]
    assert event_tables
    assert len(event_tables[0]) <= 200
    buttons = {item.label for item in app.get("download_button")}
    assert "Download historical 5W1H report" in buttons
    assert "Download complete structured training log" in buttons
    assert any(item.value == "Training validation unavailable" for item in app.info)


def _plot_specs(app: AppTest) -> dict[str, dict]:
    return {chart.key: json.loads(chart.proto.spec) for chart in app.get("plotly_chart")}


def test_semantic_colors_reach_pages_through_real_entrypoint():
    page4 = open_admin_page("4. Model Comparison")
    comparison = _plot_specs(page4)["p4_train_validation"]["data"]
    assert comparison[0]["marker"]["color"] == EVIDENCE_COLORS["validation"]
    assert comparison[1]["line"]["color"] == EVIDENCE_COLORS["training"]

    page5 = open_admin_page("5. Best Model & Importance")
    generalization = _plot_specs(page5)["p5_general_dollars"]["data"]
    assert generalization[0]["name"] == "Historical test"
    assert generalization[0]["marker"]["color"] == EVIDENCE_COLORS["historical_test"]
    assert generalization[1]["name"] == "DEV CV"
    assert generalization[1]["line"]["color"] == EVIDENCE_COLORS["development"]


@pytest.mark.parametrize(
    "label,expected_conclusion",
    [
        ("4. Model Comparison", "Candidate ranking conclusion"),
        ("5. Best Model & Importance", "Generalization conclusion"),
    ],
)
def test_report_pages_explain_question_metrics_and_evidence_conclusion(label, expected_conclusion):
    app = open_admin_page(label)
    markdown = [item.value for item in app.markdown]
    assert "**Report question**" in markdown
    assert any(value.startswith("**Current takeaway:**") for value in markdown)
    assert any(expected_conclusion in value for value in markdown)
    assert any(item.label == "How to read these metrics" for item in app.expander)


def test_page04_prioritizes_temporal_validation_guide_and_consistent_table_emphasis():
    page4 = open_admin_page("4. Model Comparison")

    expanders = [item.label for item in page4.expander]
    assert "Priority guide: How temporal validation works" in expanders
    markdown = [item.value for item in page4.markdown]
    assert "**What is trained?**" in markdown
    assert "**How are folds constructed?**" in markdown
    assert "**Why use temporal validation?**" in markdown
    assert "**Limits of this design**" in markdown
    assert any("not randomized K-fold" in value for value in markdown)
    assert any("not an expanding-window" in value for value in markdown)
    assert "**Bold = governing result** · *Italic = reference/context*" in markdown

    assert page4.table
    decision = page4.table[0].value
    assert {
        "Rank",
        "Model",
        "Decision role",
        "Mean validation MAE",
        "Mean train MAE",
        "Validation R²",
    } <= set(decision.columns)
    assert any(str(value).startswith("**") for value in decision["Model"])
    assert any(str(value).startswith("*") for value in decision["Model"])

    buttons = {button.label for button in page4.get("download_button")}
    assert "Download fold membership" in buttons
    assert "Download fold metrics" in buttons
    assert "Download complete structured training log" in buttons


def test_page05_explains_actual_tuning_and_distinguishes_applied_settings():
    page5 = open_admin_page("5. Best Model & Importance")

    assert "Priority guide: How hyperparameter tuning works" in [
        item.label for item in page5.expander
    ]
    markdown = [item.value for item in page5.markdown]
    for heading in [
        "**What is tuned?**",
        "**How does the search run?**",
        "**Why tune on temporal DEV folds?**",
        "**Selection rule**",
        "**Applied configuration**",
        "**Limitations**",
    ]:
        assert heading in markdown
    assert any("CV R²" in value and "higher is better" in value for value in markdown)
    assert any("family comparison" in value and "MAE" in value for value in markdown)
    assert any("non-nested" in value for value in markdown)
    assert any("locked test" in value for value in markdown)
    assert not any("tuning_rationale" in value for value in markdown)
    assert "**Bold = governing result** · *Italic = reference/context*" in markdown

    tuning_tables = [
        item.value
        for item in page5.table
        if {"Hyperparameter", "Stage winner", "Applied value", "Relationship"}
        <= set(item.value.columns)
    ]
    assert tuning_tables
    assert all(str(value).startswith("**") for value in tuning_tables[0]["Applied value"])

    buttons = {button.label for button in page5.get("download_button")}
    assert "Download tuning summary" in buttons
    assert "Download initial anchor grid" in buttons
    assert "Download n_estimators sweep" in buttons
    assert "Download max_depth sweep" in buttons
    assert "Download min_samples_leaf sweep" in buttons
    assert "Download max_features sweep" in buttons
    assert "Download complete structured training log" in buttons


def test_report_tables_use_reader_facing_names_and_units():
    page4 = open_admin_page("4. Model Comparison")
    page4_columns = [set(frame.value.columns) for frame in page4.dataframe]
    assert any(
        {"Fold", "Validation period", "Training rows"} <= columns for columns in page4_columns
    )

    page5 = open_admin_page("5. Best Model & Importance")
    page5_columns = [set(frame.value.columns) for frame in [*page5.table, *page5.dataframe]]
    assert any({"Metric", "Unit", "Test − CV delta"} <= columns for columns in page5_columns)
    assert any({"Feature set", "MAE (USD)", "R²"} <= columns for columns in page5_columns)


def test_prediction_conclusion_is_bound_to_current_queue_revision():
    app = open_admin_page("6. Salary Prediction")
    assert not any("Batch prediction conclusion" in item.value for item in app.markdown)
    app.button(key="p6_add").click().run()
    app.button(key="p6_run").click().run()
    assert any("Batch prediction conclusion" in item.value for item in app.markdown)
    app.button(key="p6_add").click().run()
    assert "p6_results" not in app.session_state
    assert not any("Batch prediction conclusion" in item.value for item in app.markdown)


def test_prediction_page_add_run_and_clear():
    app = open_admin_page("6. Salary Prediction")
    app.button(key="p6_add").click().run()
    assert not app.exception
    assert len(app.session_state["p6_queue"]) == 1
    app.button(key="p6_run").click().run()
    assert not app.exception
    assert app.session_state["p6_results"]["rows"]
    prediction = _plot_specs(app)["p6_prediction_interval"]["data"][0]
    assert prediction["marker"]["color"] == EVIDENCE_COLORS["prediction"]
    assert prediction["error_y"]["color"] == EVIDENCE_COLORS["uncertainty"]
    result_columns = [set(frame.value.columns) for frame in app.dataframe]
    assert any(
        {"Scenario ID", "Prediction (USD)", "Signed variance (%)"} <= columns
        for columns in result_columns
    )
    app.button(key="p6_clear").click().run()
    assert not app.exception
    assert app.session_state["p6_queue"] == []
    assert "p6_results" not in app.session_state
