from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ai_job_market.salary_inference import (
    PredictionError,
    build_prediction_audit_csv,
    build_source_schema_csv,
    predict_scenarios,
)
from ai_job_market.scenario_policy import build_scenario_policy, validate_scenarios


class Bundle:
    def __init__(self, values):
        self.values = values
        self.calls = 0

    def predict(self, frame):
        self.calls += 1
        assert list(frame.columns) == ["job_category", "years_of_experience"]
        return np.asarray(self.values[: len(frame)], dtype=float)

    def fit(self, *args, **kwargs):
        raise AssertionError("serving must not fit")


def policy():
    dev = pd.DataFrame(
        {
            "job_category": ["Research", "Research"],
            "job_title": ["Scientist", "Scientist"],
            "years_of_experience": [4.0, 10.0],
        }
    )
    meta = {"category_options": {"job_category": ["Research"], "job_title": ["Scientist"]}}
    return build_scenario_policy(dev, meta, dataset_id="dev")


def rows():
    return validate_scenarios(
        [
            {
                "scenario_id": "1",
                "source": "benchmark",
                "record_id": "r1",
                "job_category": "Research",
                "job_title": "Scientist",
                "years_of_experience": 6.0,
                "actual_salary_usd": 100000.0,
            }
        ],
        policy(),
    )


def test_prediction_math_and_audit_export_use_signed_variance():
    bundle = Bundle([110000])
    result = predict_scenarios(
        rows(),
        bundle,
        {
            "model_id": "top2",
            "evidence_id": "e1",
            "q90_abs_error_usd": 20000,
            "feature_order": ["job_category", "years_of_experience"],
        },
        policy(),
    )
    assert bundle.calls == 1
    r = result[0]
    assert r["lower_bound_usd"] == 90000
    assert r["upper_bound_usd"] == 130000
    assert r["absolute_error_usd"] == 10000
    assert r["signed_variance_pct"] == 10
    csv = build_prediction_audit_csv(result).decode()
    assert "signed_variance_pct" in csv
    assert "110000.00" in csv


def test_prediction_clips_lower_bound_and_rejects_nonfinite_output():
    result = predict_scenarios(
        rows(),
        Bundle([10000]),
        {
            "model_id": "top2",
            "evidence_id": "e1",
            "q90_abs_error_usd": 20000,
            "feature_order": ["job_category", "years_of_experience"],
        },
        policy(),
    )
    assert result[0]["lower_bound_usd"] == 0
    with pytest.raises(PredictionError, match="finite"):
        predict_scenarios(
            rows(),
            Bundle([np.nan]),
            {
                "model_id": "top2",
                "evidence_id": "e1",
                "q90_abs_error_usd": 2,
                "feature_order": ["job_category", "years_of_experience"],
            },
            policy(),
        )


def test_source_export_maps_category_to_original_column_and_leaves_salary_blank():
    data = build_source_schema_csv(rows()).decode()
    frame = pd.read_csv(pd.io.common.StringIO(data), dtype=str, keep_default_na=False)
    assert frame.loc[0, "AI Engineering"] == "Research"
    assert frame.loc[0, "annual_salary_usd"] == "100000.0"
    manual = validate_scenarios(
        [
            {
                "scenario_id": "2",
                "source": "manual",
                "job_category": "Research",
                "job_title": "Scientist",
                "years_of_experience": 7,
            }
        ],
        policy(),
    )
    frame = pd.read_csv(
        pd.io.common.StringIO(build_source_schema_csv(manual).decode()),
        dtype=str,
        keep_default_na=False,
    )
    assert frame.loc[0, "annual_salary_usd"] == ""
    assert frame.loc[0, "job_id"].startswith("SCENARIO-")


def test_wrong_model_contract_rejected_before_predict():
    bundle = Bundle([1])
    with pytest.raises(PredictionError, match="feature"):
        predict_scenarios(
            rows(),
            bundle,
            {
                "model_id": "bad",
                "evidence_id": "e1",
                "q90_abs_error_usd": 2,
                "feature_order": ["job_category"],
            },
            policy(),
        )
    assert bundle.calls == 0
