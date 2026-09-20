from __future__ import annotations

import csv
import io
import math
from typing import Any

import numpy as np
import pandas as pd

from .core import SOURCE_COLUMNS
from .scenario_policy import validate_scenarios


class PredictionError(ValueError):
    """Validated salary inference could not produce a trustworthy result."""


AUDIT_COLUMNS = [
    "schema_version",
    "scenario_id",
    "source",
    "record_id",
    "evidence_id",
    "model_id",
    "feature_variant",
    "policy_hash",
    "job_title",
    "job_category",
    "years_of_experience",
    "validation_mode",
    "exception_reason",
    "predicted_salary_usd",
    "lower_bound_usd",
    "upper_bound_usd",
    "q90_abs_error_usd",
    "q90_basis",
    "actual_salary_usd",
    "absolute_error_usd",
    "signed_variance_pct",
]


def predict_scenarios(
    rows: list[dict[str, Any]],
    bundle: Any,
    model_metadata: dict[str, Any],
    policy: dict[str, Any],
    *,
    allow_exceptions: bool = False,
    acknowledgements: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    feature_order = model_metadata.get("feature_order")
    if feature_order != ["job_category", "years_of_experience"]:
        raise PredictionError("model feature contract must be the ordered Top-2 features")
    q90 = model_metadata.get("q90_abs_error_usd")
    try:
        q90 = float(q90)
    except (TypeError, ValueError) as exc:
        raise PredictionError("q90 evidence is missing or invalid") from exc
    if not math.isfinite(q90) or q90 < 0:
        raise PredictionError("q90 evidence must be nonnegative and finite")
    validated = validate_scenarios(
        rows,
        policy,
        allow_exceptions=allow_exceptions,
        acknowledgements=acknowledgements,
    )
    if not validated:
        raise PredictionError("prediction queue is empty")
    frame = pd.DataFrame(validated)[feature_order]
    predictions = np.asarray(bundle.predict(frame), dtype=float).reshape(-1)
    if len(predictions) != len(validated) or not np.isfinite(predictions).all():
        raise PredictionError("model predictions must be finite and match the queue size")
    results: list[dict[str, Any]] = []
    for row, prediction in zip(validated, predictions):
        lower = max(0.0, float(prediction) - q90)
        upper = float(prediction) + q90
        if upper < lower:
            raise PredictionError("prediction interval ordering is invalid")
        actual = row.get("actual_salary_usd")
        absolute_error = None
        signed_variance = None
        if actual not in (None, ""):
            if isinstance(actual, bool):
                raise PredictionError("actual salary must be a nonnegative finite number")
            try:
                actual = float(actual)
            except (TypeError, ValueError) as exc:
                raise PredictionError("actual salary must be a nonnegative finite number") from exc
            if not math.isfinite(actual) or actual < 0:
                raise PredictionError("actual salary must be a nonnegative finite number")
            absolute_error = abs(float(prediction) - actual)
            if actual > 0:
                signed_variance = 100.0 * (float(prediction) - actual) / actual
        result = dict(row)
        result.update(
            {
                "schema_version": 1,
                "evidence_id": str(model_metadata.get("evidence_id", "")),
                "model_id": str(model_metadata.get("model_id", "")),
                "feature_variant": "top2",
                "predicted_salary_usd": float(prediction),
                "lower_bound_usd": lower,
                "upper_bound_usd": upper,
                "q90_abs_error_usd": q90,
                "q90_basis": str(
                    model_metadata.get("q90_basis", "historical_test_absolute_errors")
                ),
                "actual_salary_usd": actual,
                "absolute_error_usd": absolute_error,
                "signed_variance_pct": signed_variance,
            }
        )
        results.append(result)
    return results


def _format(value: Any, decimals: int) -> str:
    if value is None or value == "" or (isinstance(value, float) and math.isnan(value)):
        return ""
    return f"{float(value):.{decimals}f}"


def build_prediction_audit_csv(rows: list[dict[str, Any]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=AUDIT_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    for source in rows:
        row = {column: source.get(column, "") for column in AUDIT_COLUMNS}
        row["schema_version"] = 1
        for column in [
            "predicted_salary_usd",
            "lower_bound_usd",
            "upper_bound_usd",
            "q90_abs_error_usd",
            "actual_salary_usd",
            "absolute_error_usd",
        ]:
            row[column] = _format(row[column], 2)
        row["signed_variance_pct"] = _format(row["signed_variance_pct"], 4)
        writer.writerow(row)
    return buffer.getvalue().encode("utf-8")


def build_source_schema_csv(rows: list[dict[str, Any]]) -> bytes:
    output: list[dict[str, Any]] = []
    for source in rows:
        row = {column: "" for column in SOURCE_COLUMNS}
        row["job_id"] = f"SCENARIO-{source.get('scenario_id', '')}"
        row["job_title"] = source.get("job_title", "")
        row["AI Engineering"] = source.get("job_category", "")
        row["years_of_experience"] = source.get("years_of_experience", "")
        if source.get("source") == "benchmark":
            for column in SOURCE_COLUMNS:
                if column in source and source[column] is not None:
                    row[column] = source[column]
            row["AI Engineering"] = source.get("job_category", row["AI Engineering"])
            if source.get("actual_salary_usd") not in (None, ""):
                row["annual_salary_usd"] = source["actual_salary_usd"]
            row["job_id"] = f"SCENARIO-{source.get('scenario_id', '')}"
        output.append(row)
    return pd.DataFrame(output, columns=SOURCE_COLUMNS).to_csv(index=False).encode("utf-8")


def build_growth_inputs(
    rows: list[dict[str, Any]],
    policy: dict[str, Any],
    *,
    extend: bool = False,
    acknowledgement: bool = False,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    profiles = {(r["job_category"], r["job_title"]) for r in rows}
    for category, title in sorted(profiles):
        bounds = policy["titles"][title]
        values = list(range(math.ceil(bounds["min"]), math.floor(bounds["max"]) + 1))
        values.extend([float(bounds["min"]), float(bounds["max"])])
        if extend:
            if not acknowledgement:
                raise PredictionError("acknowledgement required for extended growth curve")
            values.extend(range(0, 16))
        for years in sorted(set(map(float, values))):
            outside = not float(bounds["min"]) <= years <= float(bounds["max"])
            result.append(
                {
                    "scenario_id": f"curve-{category}-{title}-{years:g}",
                    "source": "benchmark" if outside else "manual",
                    "job_category": category,
                    "job_title": title,
                    "years_of_experience": years,
                    "validation_mode": "experience_exception" if outside else "strict",
                    "exception_reason": "experience_outside_observed_dev_range" if outside else "",
                }
            )
    return result
