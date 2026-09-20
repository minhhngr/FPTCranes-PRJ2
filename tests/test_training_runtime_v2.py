from __future__ import annotations

import numpy as np
import pandas as pd

from ai_job_market.training_runtime import (
    assess_operational_budgets,
    benchmark_bundle_load,
    benchmark_prediction,
    compute_accuracy_fit_pareto,
    summarize_runtime_samples,
    timing_accounting,
)


class FakeClock:
    def __init__(self, step: int) -> None:
        self.value = 0
        self.step = step

    def __call__(self) -> int:
        current = self.value
        self.value += self.step
        return current


class RecordingPredictor:
    def __init__(self) -> None:
        self.batch_sizes = []

    def predict(self, frame):
        self.batch_sizes.append(len(frame))
        return np.arange(len(frame), dtype=float)


def test_benchmark_has_three_warmups_thirty_samples_and_exact_batches() -> None:
    predictor = RecordingPredictor()
    frame = pd.DataFrame({"feature": range(10)})
    samples, metadata = benchmark_prediction(
        predictor,
        frame,
        model_id="model-1",
        fold_id="outer-1",
        environment_id="env-1",
        batch_sizes=[1, 100],
        warmups=3,
        measured_calls=30,
        wall_clock_ns=FakeClock(1_000_000),
        cpu_clock_ns=FakeClock(500_000),
    )
    assert len(samples) == 60
    assert predictor.batch_sizes.count(1) == 33
    assert predictor.batch_sizes.count(10) == 33
    assert set(samples["batch_size"]) == {1, 10}
    assert samples["wall_ns"].eq(1_000_000).all()
    assert samples["cpu_ns"].eq(500_000).all()
    assert metadata["warmup_calls"] == 6
    assert metadata["measured_calls"] == 60
    assert metadata["benchmark_predict_calls"] == 66


def test_runtime_summary_reproduces_linear_quantiles_and_throughput() -> None:
    samples = pd.DataFrame(
        {
            "model_id": ["m"] * 4,
            "fold_id": ["f"] * 4,
            "environment_id": ["e"] * 4,
            "method_version": ["v"] * 4,
            "batch_size": [2] * 4,
            "repeat": [1, 2, 3, 4],
            "wall_ns": [1_000_000, 2_000_000, 3_000_000, 4_000_000],
            "cpu_ns": [500_000] * 4,
        }
    )
    summary = summarize_runtime_samples(samples).iloc[0]
    assert summary["p50_ms"] == 2.5
    assert summary["p90_ms"] == 3.7
    assert summary["sample_count"] == 4
    assert summary["throughput_rows_per_s"] == 800.0
    assert summary["amortized_ms_per_row"] == 1.25


def test_parent_child_timing_accounting_exposes_overhead() -> None:
    accounting = timing_accounting(
        parent_wall_s=10.0,
        parent_cpu_s=8.0,
        child_wall_s=[2.0, 3.0],
        child_cpu_s=[1.0, 2.0],
    )
    assert accounting["child_wall_s"] == 5.0
    assert accounting["overhead_wall_s"] == 5.0
    assert accounting["overhead_cpu_s"] == 5.0


def test_pareto_uses_unrounded_accuracy_and_fit_cost() -> None:
    evidence = pd.DataFrame(
        [
            {"model": "A", "cv_mae_mean_usd": 10.0, "total_cv_fit_wall_s": 5.0},
            {"model": "B", "cv_mae_mean_usd": 11.0, "total_cv_fit_wall_s": 4.0},
            {"model": "C", "cv_mae_mean_usd": 12.0, "total_cv_fit_wall_s": 6.0},
        ]
    )
    result = compute_accuracy_fit_pareto(evidence)
    assert set(result.loc[result["pareto"], "model"]) == {"A", "B"}
    assert result.loc[result["model"] == "C", "dominated_by"].iloc[0] == ["A", "B"]


def test_bundle_measurement_separates_first_and_repeated_loads(tmp_path) -> None:
    bundle = tmp_path / "model.joblib"
    bundle.write_bytes(b"bundle")
    loads = []

    def loader(path):
        loads.append(path)
        return {"loaded": True}

    evidence, samples = benchmark_bundle_load(
        bundle,
        loader=loader,
        repeated_loads=3,
        wall_clock_ns=FakeClock(2_000_000),
        rss_reader=lambda: 100,
    )
    assert evidence["bundle_bytes"] == 6
    assert len(evidence["bundle_sha256"]) == 64
    assert evidence["first_load_ms"] == 2.0
    assert len(samples) == 3
    assert samples["load_ms"].eq(2.0).all()
    assert len(loads) == 4
    assert evidence["rss_delta_bytes"] == 0


def test_operational_budgets_do_not_default_to_pass() -> None:
    observed = {
        "run_wall_s": 10.0,
        "final_full_batch1_p95_ms": 2.0,
        "final_top2_batch1_p95_ms": 1.0,
        "full_bundle_bytes": 1000,
        "top2_bundle_bytes": 500,
    }
    no_limits = assess_operational_budgets(observed, limits={}, scientific_outcome="Good")
    assert no_limits["operational_status"] == "not_assessed"
    assert no_limits["deployment_review_eligible"] is False

    limits = {
        "max_run_wall_s": 20.0,
        "max_final_batch1_p95_ms": 1.5,
        "max_bundle_bytes": 2000,
    }
    assessed = assess_operational_budgets(observed, limits=limits, scientific_outcome="Good")
    assert assessed["operational_status"] == "fail"
    assert assessed["criteria"]["final_full_batch1_p95_ms"]["verdict"] == "fail"
    assert assessed["deployment_review_eligible"] is False


def test_declared_global_prediction_call_ceiling_is_1784() -> None:
    contexts = 25 + 2
    assert contexts * 2 * (3 + 30) + 2 == 1784
