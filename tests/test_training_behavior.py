import math
from io import StringIO

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

from ai_job_market import core


def _dev_frame(n: int = 15) -> pd.DataFrame:
    rows = []
    for i in range(n):
        rows.append(
            {
                "posting_year": 2025 + (i // 12),
                "posting_month": (i % 12) + 1,
                "years_of_experience": float(i),
                core.TARGET: float(i * 1000 + 50000),
            }
        )
    return pd.DataFrame(rows)


def test_temporal_cv_splits_use_stable_positional_blocks_and_remainder():
    dev = _dev_frame(15)

    splits = core.temporal_cv_splits(dev, n_splits=4, block_size=3)

    assert len(splits) == 4
    assert [tr.tolist() for tr, _, _ in splits] == [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
        [9, 10, 11],
    ]
    assert [va.tolist() for _, va, _ in splits] == [
        [3, 4, 5],
        [6, 7, 8],
        [9, 10, 11],
        [12, 13, 14],
    ]
    assert [len(va) for _, va, _ in splits] == [3, 3, 3, 3]


class CountingRegressor(BaseEstimator, RegressorMixin):
    fit_calls = 0
    predict_calls = 0

    def fit(self, X, y):
        type(self).fit_calls += 1
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        type(self).predict_calls += 1
        return np.full(len(X), self.mean_, dtype=float)


def test_evaluate_model_cv_preserves_one_fit_and_predict_per_fold():
    CountingRegressor.fit_calls = 0
    CountingRegressor.predict_calls = 0
    dev = _dev_frame(800)

    fold, summary = core.evaluate_model_cv(
        dev,
        ["years_of_experience"],
        "Counting",
        CountingRegressor(),
        n_splits=3,
    )

    assert summary["folds"] == len(fold) == 3
    assert CountingRegressor.fit_calls == 3
    assert CountingRegressor.predict_calls == 3
    assert set(["MAE", "RMSE", "R2", "MedAE"]).issubset(fold.columns)


def test_manual_rf_tuning_keeps_r2_ranking_and_fixed_depth(monkeypatch):
    calls = []

    def fake_evaluate_model_cv(dev, features, model_name, model, n_splits):
        params = model.get_params(deep=False)
        calls.append((model_name, params))
        score = float(len(calls)) / 100.0
        return pd.DataFrame(), {
            "model": model_name,
            "R2_mean": score,
            "MAE_mean": 1000.0 - score,
            "MAE_std": score,
            "RMSE_mean": 2000.0 - score,
            "MedAE_mean": 500.0 - score,
        }

    monkeypatch.setattr(core, "evaluate_model_cv", fake_evaluate_model_cv)

    df1, df2, df3, df4, df5, summary = core.tune_random_forest_manual_steps(
        _dev_frame(), n_splits=5, seed=7
    )

    assert df1.CV_R2.is_monotonic_decreasing
    assert df2.CV_R2.is_monotonic_decreasing
    assert df3.CV_R2.is_monotonic_decreasing
    assert df4.CV_R2.is_monotonic_decreasing
    assert df5.CV_R2.is_monotonic_decreasing
    assert summary.loc[summary.hyperparameter == "2. max_depth", "optimal_value"].iloc[0] == "20"
    assert any(name.startswith("RF depth_") for name, _ in calls)


def test_debug_and_normal_observation_preserve_cv_results_and_call_counts(tmp_path):
    dev = _dev_frame(800)
    results = []
    for mode in [None, False, True]:
        CountingRegressor.fit_calls = 0
        CountingRegressor.predict_calls = 0
        if mode is None:
            fold, summary = core.evaluate_model_cv(
                dev, ["years_of_experience"], "Counting", CountingRegressor(), n_splits=2
            )
        else:
            with core.start_training_audit(
                workspace_root=tmp_path / str(mode),
                raw_path=tmp_path / "input.csv",
                seed=42,
                target=core.TARGET,
                features=["years_of_experience"],
                stream=StringIO(),
                debuglog=mode,
            ):
                fold, summary = core.evaluate_model_cv(
                    dev, ["years_of_experience"], "Counting", CountingRegressor(), n_splits=2
                )
        results.append((fold.drop(columns=["fit_time_s", "predict_time_s"]), summary))
        assert CountingRegressor.fit_calls == 2
        assert CountingRegressor.predict_calls == 2

    for fold, summary in results[1:]:
        pd.testing.assert_frame_equal(fold, results[0][0])
        for metric in ["MAE_mean", "MAE_std", "RMSE_mean", "R2_mean", "MedAE_mean", "folds"]:
            assert summary[metric] == results[0][1][metric]


def test_final_model_selection_consumes_passed_tuning_table_not_later_stage_assumption():
    tuning = pd.DataFrame(
        [
            {
                "n_estimators": 123,
                "min_samples_leaf": 4,
                "max_features": 0.6,
                "max_depth": math.nan,
            }
        ]
    )

    model = core.final_model_from_selection("Random Forest", tuning, seed=99)

    assert model.n_estimators == 123
    assert model.min_samples_leaf == 4
    assert model.max_features == 0.6
    assert model.max_depth is None
    assert model.random_state == 99
