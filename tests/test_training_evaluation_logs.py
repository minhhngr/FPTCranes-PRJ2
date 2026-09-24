import json
from io import StringIO

import pandas as pd

from ai_job_market import core


def _dev_frame(n=800):
    return pd.DataFrame(
        {
            "posting_year": [2025 + i // 400 for i in range(n)],
            "posting_month": [(i % 12) + 1 for i in range(n)],
            "years_of_experience": [float(i) for i in range(n)],
            core.TARGET: [50_000.0 + i for i in range(n)],
        }
    )


def _events(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_evaluation_completed_records_all_metrics_and_honest_fit_status(tmp_path):
    with core.start_training_audit(
        workspace_root=tmp_path,
        raw_path=tmp_path / "input.csv",
        seed=42,
        target=core.TARGET,
        features=["years_of_experience"],
        stream=StringIO(),
    ) as audit:
        fold, summary = core.evaluate_model_cv(
            _dev_frame(),
            ["years_of_experience"],
            "Linear Regression",
            core.LinearRegression(),
            n_splits=2,
        )
        log_path = audit.log_path

    events = _events(log_path)
    completed = [e for e in events if e["event"] == "evaluation_completed"]
    assert len(completed) == 1
    event = completed[0]
    assert event["model_name"] == "Linear Regression"
    assert event["folds"] == len(fold) == summary["folds"]
    for name in ["MAE_mean", "MAE_std", "RMSE_mean", "R2_mean", "MedAE_mean"]:
        assert event[name] == summary[name]
    assert event["fit_assessment"] == "insufficient_evidence"
    assert event["fit_reason"] == "training_scores_not_computed; diagnostic_rule_not_defined"
    assert event["train_metrics_status"] == "not_computed"
    assert event["locked_test_metrics_status"] == "not_evaluated"


def test_manual_rf_tuning_records_all_55_trials_without_changing_rankings(monkeypatch, tmp_path):
    calls = []

    def fake_evaluate(dev, features, model_name, model, n_splits):
        calls.append(model.get_params(deep=False))
        score = len(calls) / 100
        return pd.DataFrame(), {
            "model": model_name,
            "R2_mean": score,
            "MAE_mean": 1000 - score,
            "MAE_std": score,
            "RMSE_mean": 2000 - score,
            "MedAE_mean": 500 - score,
        }

    monkeypatch.setattr(core, "evaluate_model_cv", fake_evaluate)
    with core.start_training_audit(
        workspace_root=tmp_path,
        raw_path=tmp_path / "input.csv",
        seed=7,
        target=core.TARGET,
        features=core.MODEL_FEATURES,
        stream=StringIO(),
    ) as audit:
        tables = core.tune_random_forest_manual_steps(_dev_frame(20), n_splits=2, seed=7)
        log_path = audit.log_path

    events = _events(log_path)
    assert len([e for e in events if e["event"] == "trial_started"]) == 55
    assert len([e for e in events if e["event"] == "trial_completed"]) == 55
    assert len(calls) == 55
    assert all(table.CV_R2.is_monotonic_decreasing for table in tables[:5])


def test_non_random_forest_selection_can_record_tuning_skip_without_changing_model(tmp_path):
    with core.start_training_audit(
        workspace_root=tmp_path,
        raw_path=tmp_path / "input.csv",
        seed=7,
        target=core.TARGET,
        features=core.MODEL_FEATURES,
        stream=StringIO(),
    ) as audit:
        model = core.final_model_from_selection("Ridge Regression", pd.DataFrame(), seed=7)
        core.emit_event(
            "tuning_skipped",
            step_id="tuning.rf",
            operation="manual_rf_tuning",
            status="skipped",
            message="Random Forest tuning skipped because another family won.",
            extra={"best_family": "Ridge Regression", "reason": "random_forest_not_selected"},
        )
        log_path = audit.log_path

    assert model.alpha == 10.0
    assert any(e["event"] == "tuning_skipped" for e in _events(log_path))
