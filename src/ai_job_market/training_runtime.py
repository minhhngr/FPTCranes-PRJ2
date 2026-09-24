"""Bounded runtime measurement and operational assessment helpers."""

from __future__ import annotations

import hashlib
import platform
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

Clock = Callable[[], int]


def benchmark_prediction(
    predictor: Any,
    training_features: pd.DataFrame,
    *,
    model_id: str,
    fold_id: str,
    environment_id: str,
    batch_sizes: list[int],
    warmups: int,
    measured_calls: int,
    wall_clock_ns: Clock = time.perf_counter_ns,
    cpu_clock_ns: Clock = time.process_time_ns,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if training_features.empty:
        raise ValueError("prediction benchmark requires TRAIN feature rows")
    if warmups < 0 or measured_calls <= 0:
        raise ValueError("prediction benchmark counts are invalid")
    actual_batches = sorted({min(int(size), len(training_features)) for size in batch_sizes})
    if any(size <= 0 or size > 100 for size in actual_batches):
        raise ValueError("prediction benchmark batch size must be within 1..100")
    rows: list[dict[str, Any]] = []
    warmup_calls = 0
    for batch_size in actual_batches:
        batch = training_features.iloc[:batch_size]
        for _ in range(warmups):
            predictor.predict(batch)
            warmup_calls += 1
        for repeat in range(1, measured_calls + 1):
            wall_start = wall_clock_ns()
            cpu_start = cpu_clock_ns()
            predictor.predict(batch)
            cpu_elapsed = cpu_clock_ns() - cpu_start
            wall_elapsed = wall_clock_ns() - wall_start
            if wall_elapsed <= 0 or cpu_elapsed < 0:
                raise ValueError("benchmark clock resolution is insufficient")
            rows.append(
                {
                    "model_id": model_id,
                    "fold_id": fold_id,
                    "environment_id": environment_id,
                    "method_version": "warm-train-predict/v1",
                    "batch_size": batch_size,
                    "repeat": repeat,
                    "wall_ns": int(wall_elapsed),
                    "cpu_ns": int(cpu_elapsed),
                }
            )
    measured = len(rows)
    return pd.DataFrame(rows), {
        "method_version": "warm-train-predict/v1",
        "warmup_calls": warmup_calls,
        "measured_calls": measured,
        "benchmark_predict_calls": warmup_calls + measured,
        "batch_sizes": actual_batches,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cache_state": "uncontrolled",
    }


def _max_rss_bytes() -> int:
    try:
        import resource

        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value * 1024)  # Linux reports KiB.
    except ModuleNotFoundError:
        return 0


def benchmark_bundle_load(
    bundle_path: Path,
    *,
    loader: Callable[[Path], Any],
    repeated_loads: int = 30,
    wall_clock_ns: Clock = time.perf_counter_ns,
    rss_reader: Callable[[], int] = _max_rss_bytes,
) -> tuple[dict[str, Any], pd.DataFrame]:
    if repeated_loads <= 0 or not bundle_path.is_file() or bundle_path.is_symlink():
        raise ValueError("bundle load benchmark input is invalid")
    digest = hashlib.sha256(bundle_path.read_bytes()).hexdigest()
    rss_before = rss_reader()
    started = wall_clock_ns()
    loaded = loader(bundle_path)
    first_load_ns = wall_clock_ns() - started
    rss_after = rss_reader()
    del loaded
    rows: list[dict[str, Any]] = []
    for repeat in range(1, repeated_loads + 1):
        started = wall_clock_ns()
        loaded = loader(bundle_path)
        elapsed = wall_clock_ns() - started
        del loaded
        if elapsed <= 0:
            raise ValueError("bundle load clock resolution is insufficient")
        rows.append({"repeat": repeat, "load_ns": elapsed, "load_ms": elapsed / 1_000_000})
    evidence = {
        "bundle_path": bundle_path.name,
        "bundle_bytes": bundle_path.stat().st_size,
        "bundle_sha256": digest,
        "first_load_ms": first_load_ns / 1_000_000,
        "repeated_load_count": repeated_loads,
        "rss_before_bytes": rss_before,
        "rss_after_first_load_bytes": rss_after,
        "rss_delta_bytes": rss_after - rss_before,
        "cache_state": "same_local_filesystem_uncontrolled_os_cache",
    }
    return evidence, pd.DataFrame(rows)


def summarize_runtime_samples(samples: pd.DataFrame) -> pd.DataFrame:
    required = {
        "model_id",
        "fold_id",
        "environment_id",
        "method_version",
        "batch_size",
        "wall_ns",
        "cpu_ns",
    }
    missing = sorted(required - set(samples.columns))
    if missing:
        raise ValueError(f"runtime samples are missing fields: {missing}")
    rows: list[dict[str, Any]] = []
    groups = ["model_id", "fold_id", "environment_id", "method_version", "batch_size"]
    for identity, group in samples.groupby(groups, sort=False, dropna=False):
        wall_ms = group["wall_ns"].to_numpy(dtype=float) / 1_000_000
        total_seconds = float(group["wall_ns"].sum()) / 1_000_000_000
        sample_count = len(group)
        batch_size = int(identity[-1])
        rows.append(
            {
                **dict(zip(groups, identity, strict=True)),
                "sample_count": sample_count,
                "min_ms": float(np.min(wall_ms)),
                "p50_ms": float(np.quantile(wall_ms, 0.5, method="linear")),
                "p90_ms": float(np.quantile(wall_ms, 0.9, method="linear")),
                "p95_ms": float(np.quantile(wall_ms, 0.95, method="linear")),
                "max_ms": float(np.max(wall_ms)),
                "mean_ms": float(np.mean(wall_ms)),
                "population_sd_ms": float(np.std(wall_ms, ddof=0)),
                "cpu_mean_ms": float(group["cpu_ns"].mean() / 1_000_000),
                "throughput_rows_per_s": batch_size * sample_count / total_seconds,
                "amortized_ms_per_row": total_seconds * 1000 / (batch_size * sample_count),
            }
        )
    return pd.DataFrame(rows)


def timing_accounting(
    *,
    parent_wall_s: float,
    parent_cpu_s: float,
    child_wall_s: list[float],
    child_cpu_s: list[float],
) -> dict[str, float]:
    if len(child_wall_s) != len(child_cpu_s):
        raise ValueError("wall/CPU child timing counts differ")
    values = [parent_wall_s, parent_cpu_s, *child_wall_s, *child_cpu_s]
    if not np.isfinite(values).all() or any(value < 0 for value in values):
        raise ValueError("timing accounting requires finite nonnegative values")
    wall_total = float(sum(child_wall_s))
    cpu_total = float(sum(child_cpu_s))
    if wall_total > parent_wall_s + 1e-9 or cpu_total > parent_cpu_s + 1e-9:
        raise ValueError("sequential child time exceeds parent time")
    return {
        "parent_wall_s": float(parent_wall_s),
        "parent_cpu_s": float(parent_cpu_s),
        "child_wall_s": wall_total,
        "child_cpu_s": cpu_total,
        "overhead_wall_s": float(parent_wall_s - wall_total),
        "overhead_cpu_s": float(parent_cpu_s - cpu_total),
    }


def compute_accuracy_fit_pareto(evidence: pd.DataFrame) -> pd.DataFrame:
    required = {"model", "cv_mae_mean_usd", "total_cv_fit_wall_s"}
    if required - set(evidence.columns):
        raise ValueError("Pareto evidence is incomplete")
    out = evidence.copy()
    dominated_by: list[list[str]] = []
    for _, candidate in out.iterrows():
        dominators: list[str] = []
        for _, other in out.iterrows():
            if candidate.model == other.model:
                continue
            no_worse = float(other.cv_mae_mean_usd) <= float(candidate.cv_mae_mean_usd) and float(
                other.total_cv_fit_wall_s
            ) <= float(candidate.total_cv_fit_wall_s)
            strictly_better = float(other.cv_mae_mean_usd) < float(
                candidate.cv_mae_mean_usd
            ) or float(other.total_cv_fit_wall_s) < float(candidate.total_cv_fit_wall_s)
            if no_worse and strictly_better:
                dominators.append(str(other.model))
        dominated_by.append(dominators)
    out["dominated_by"] = dominated_by
    out["pareto"] = out["dominated_by"].map(lambda values: len(values) == 0)
    return out


def assess_operational_budgets(
    observed: dict[str, float | int | None],
    *,
    limits: dict[str, float | int],
    scientific_outcome: str,
) -> dict[str, Any]:
    mapping = {
        "run_wall_s": "max_run_wall_s",
        "final_full_batch1_p95_ms": "max_final_batch1_p95_ms",
        "final_top2_batch1_p95_ms": "max_final_batch1_p95_ms",
        "full_bundle_bytes": "max_bundle_bytes",
        "top2_bundle_bytes": "max_bundle_bytes",
    }
    if not limits:
        return {
            "operational_status": "not_assessed",
            "criteria": {
                name: {"verdict": "not_configured", "observed": observed.get(name)}
                for name in mapping
            },
            "deployment_review_eligible": False,
        }
    criteria: dict[str, dict[str, Any]] = {}
    for name, limit_name in mapping.items():
        if limit_name not in limits:
            verdict = "not_configured"
        elif observed.get(name) is None:
            verdict = "unavailable"
        else:
            verdict = "pass" if float(observed[name]) <= float(limits[limit_name]) else "fail"
        criteria[name] = {
            "verdict": verdict,
            "observed": observed.get(name),
            "limit": limits.get(limit_name),
            "limit_name": limit_name,
        }
    verdicts = {criterion["verdict"] for criterion in criteria.values()}
    if "fail" in verdicts:
        status = "fail"
    elif verdicts == {"pass"}:
        status = "pass"
    elif "unavailable" in verdicts:
        status = "unavailable"
    else:
        status = "not_assessed"
    return {
        "operational_status": status,
        "criteria": criteria,
        "deployment_review_eligible": scientific_outcome == "Good" and status == "pass",
    }
