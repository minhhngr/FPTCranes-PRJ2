import json
from io import StringIO

import pandas as pd

from ai_job_market import core


def _events(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _frame(n: int) -> pd.DataFrame:
    years = [2025 + (i // 24) for i in range(n)]
    months = [((i // 2) % 12) + 1 for i in range(n)]
    return pd.DataFrame(
        {
            "posting_year": years,
            "posting_month": months,
            "years_of_experience": [float(i) for i in range(n)],
            core.TARGET: [float(i * 10) for i in range(n)],
        }
    )


def test_fold_audit_records_actual_membership_bounds_and_overlap(tmp_path):
    dev = _frame(800)

    with core.start_training_audit(
        workspace_root=tmp_path,
        raw_path=tmp_path / "input.csv",
        seed=42,
        target=core.TARGET,
        features=["years_of_experience"],
        stream=StringIO(),
    ) as audit:
        core.evaluate_model_cv(
            dev,
            ["years_of_experience"],
            "Linear Regression",
            core.LinearRegression(),
            n_splits=3,
        )
        log_path = audit.log_path

    split_events = [event for event in _events(log_path) if event["event"] == "split_defined"]
    assert len(split_events) == 1
    split = split_events[0]
    assert split["split_id"]
    assert split["dataset_row_order_sha256"]
    assert split["sort_policy"] == "posting_year, posting_month stable mergesort"
    assert split["requested_folds"] == 3
    assert split["effective_folds"] == 3
    first_fold = split["folds"][0]
    assert first_fold["train_positions"] == list(range(200))
    assert first_fold["validation_positions"] == list(range(200, 400))
    assert first_fold["row_overlap_count"] == 0
    assert first_fold["shared_periods"] == []
    assert first_fold["train_period_min"] == "2025-01"
    assert first_fold["validation_period_max"] >= first_fold["validation_period_min"]


def test_importance_fold_audit_reports_metrics_not_computed(tmp_path):
    dev = _frame(450)

    with core.start_training_audit(
        workspace_root=tmp_path,
        raw_path=tmp_path / "input.csv",
        seed=42,
        target=core.TARGET,
        features=core.MODEL_FEATURES,
        stream=StringIO(),
    ) as audit:
        # Use all required production columns with compatible simple values.
        for col in core.MODEL_FEATURES:
            if col not in dev:
                dev[col] = "x"
        for col in ["demand_score", "benefits_score_10", "skill_count"]:
            dev[col] = 1.0
        dev["required_skills"] = "python|sql"
        core.random_forest_importance_by_fold(dev, seed=1, n_splits=1)
        log_path = audit.log_path

    events = _events(log_path)
    assert any(event["event"] == "split_defined" and event["metrics_status"] == "not_computed" for event in events)
    assert any(event["operation"] == "extract_encoded_importance" for event in events)


def test_dataset_identity_changes_for_same_sized_different_content():
    first = _frame(20)
    second = first.copy()
    second.loc[0, core.TARGET] += 1

    first_id = core._temporal_fold_audit_payload(
        first, core.temporal_cv_splits(first, n_splits=1, block_size=5), 1
    )["dataset_id"]
    second_id = core._temporal_fold_audit_payload(
        second, core.temporal_cv_splits(second, n_splits=1, block_size=5), 1
    )["dataset_id"]

    assert first_id != second_id
