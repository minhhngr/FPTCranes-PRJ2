from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression

from ai_job_market.ui_evidence_training import (
    evaluate_frozen_models,
    select_benchmark_offsets,
)


def frame():
    return pd.DataFrame(
        {
            "job_category": ["A", "A", "B", "B", "A", "B"],
            "years_of_experience": [1, 2, 3, 4, 5, 6],
            "annual_salary_usd": [10, 20, 30, 40, 50, 60],
            "posting_year": [2025] * 6,
            "posting_month": [1, 1, 2, 2, 3, 3],
        }
    )


def test_evaluator_scores_train_and_validation_from_shared_fits():
    data = frame()
    splits = [(np.array([0, 1, 2]), np.array([3, 4]), "2025-02..2025-03")]
    folds, summary, _ = evaluate_frozen_models(
        data,
        ["job_category", "years_of_experience"],
        {"Dummy": DummyRegressor(strategy="median"), "Linear": LinearRegression()},
        splits,
        capture_importance=False,
    )
    assert len(folds) == 2
    assert set(folds.columns) >= {"train_MAE", "validation_MAE", "train_R2", "validation_R2"}
    assert (folds.train_rows == 3).all()
    assert (folds.validation_rows == 2).all()
    assert set(summary.model) == {"Dummy", "Linear"}
    assert (summary.effective_folds == 1).all()


def test_benchmark_selection_uses_source_order_not_targets_or_errors():
    test = frame().iloc[[5, 1, 4, 2]].reset_index(drop=True)
    eligible = pd.Series([True, False, True, True])
    assert select_benchmark_offsets(test, eligible, limit=3) == [0, 2, 3]
    mutated = test.assign(annual_salary_usd=[9999, -1, 100000, 0])
    assert select_benchmark_offsets(mutated, eligible, limit=3) == [0, 2, 3]


def test_benchmark_selection_is_repeatable_when_candidates_exceed_limit():
    test = frame()
    eligible = pd.Series([True, True, True, True, True, False])
    selected = select_benchmark_offsets(test, eligible, limit=3)
    assert selected == [0, 2, 3]
    assert select_benchmark_offsets(test, eligible, limit=3) == selected
