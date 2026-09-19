from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import pandas as pd


class ScenarioValidationError(ValueError):
    """A scenario or policy failed the declared serving contract."""


def _digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def build_scenario_policy(
    dev: pd.DataFrame, metadata: dict[str, Any], *, dataset_id: str
) -> dict[str, Any]:
    required = {"job_category", "job_title", "years_of_experience"}
    missing = required - set(dev.columns)
    if missing:
        raise ScenarioValidationError(f"development data missing columns: {sorted(missing)}")
    options = metadata.get("category_options", {})
    allowed_categories = set(map(str, options.get("job_category", [])))
    allowed_titles = set(map(str, options.get("job_title", [])))
    observed_categories = set(dev["job_category"].dropna().astype(str))
    observed_titles = set(dev["job_title"].dropna().astype(str))
    if not observed_categories <= allowed_categories or not observed_titles <= allowed_titles:
        raise ScenarioValidationError("metadata enums do not cover observed development values")

    work = dev[list(required)].copy()
    work["job_category"] = work["job_category"].astype(str)
    work["job_title"] = work["job_title"].astype(str)
    work["years_of_experience"] = pd.to_numeric(work["years_of_experience"], errors="coerce")
    if not work["years_of_experience"].map(math.isfinite).all():
        raise ScenarioValidationError("development experience values must be finite")

    pairs = (
        work.groupby(["job_category", "job_title"], observed=True)
        .size()
        .rename("support")
        .reset_index()
        .sort_values(["job_category", "job_title"])
        .to_dict("records")
    )
    titles: dict[str, dict[str, float | int]] = {}
    for title, group in work.groupby("job_title", observed=True):
        values = group["years_of_experience"]
        titles[str(title)] = {
            "min": float(values.min()),
            "max": float(values.max()),
            "median": float(values.median()),
            "support": int(len(values)),
        }
    body: dict[str, Any] = {
        "schema_version": 1,
        "dataset_id": str(dataset_id),
        "pair_source": "observed_dev",
        "pairs": pairs,
        "titles": dict(sorted(titles.items())),
        "exploratory_experience_min": 0.0,
        "exploratory_experience_max": 15.0,
        "max_queue": 100,
    }
    body["policy_hash"] = _digest(body)
    return body


def _finite_years(value: Any) -> float:
    if isinstance(value, bool):
        raise ScenarioValidationError("years_of_experience must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ScenarioValidationError("years_of_experience must be a finite number") from exc
    if not math.isfinite(result):
        raise ScenarioValidationError("years_of_experience must be a finite number")
    return result


def validate_scenarios(
    rows: list[dict[str, Any]],
    policy: dict[str, Any],
    *,
    allow_exceptions: bool = False,
    acknowledgements: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    if len(rows) > int(policy.get("max_queue", 100)):
        raise ScenarioValidationError("scenario queue exceeds the 100-row limit")
    pairs = {(p["job_category"], p["job_title"]) for p in policy.get("pairs", [])}
    titles = policy.get("titles", {})
    acknowledgements = acknowledgements or {}
    validated: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, source_row in enumerate(rows, start=1):
        row = dict(source_row)
        scenario_id = str(row.get("scenario_id") or index)
        if scenario_id in seen_ids:
            raise ScenarioValidationError(f"duplicate scenario_id: {scenario_id}")
        seen_ids.add(scenario_id)
        category = str(row.get("job_category", ""))
        title = str(row.get("job_title", ""))
        if (category, title) not in pairs:
            raise ScenarioValidationError(
                f"scenario {scenario_id}: category/title pair was not observed in DEV"
            )
        years = _finite_years(row.get("years_of_experience"))
        bounds = titles.get(title)
        if not bounds:
            raise ScenarioValidationError(f"scenario {scenario_id}: title has no DEV bounds")
        in_bounds = float(bounds["min"]) <= years <= float(bounds["max"])
        mode = "strict"
        reason = ""
        if not in_bounds:
            can_except = (
                allow_exceptions
                and row.get("source") == "benchmark"
                and float(policy["exploratory_experience_min"])
                <= years
                <= float(policy["exploratory_experience_max"])
            )
            if not can_except:
                raise ScenarioValidationError(
                    f"scenario {scenario_id}: experience is outside observed DEV bounds "
                    f"[{bounds['min']}, {bounds['max']}]"
                )
            if acknowledgements.get(scenario_id) != policy.get("policy_hash"):
                raise ScenarioValidationError(
                    f"scenario {scenario_id}: acknowledgement required for experience exception"
                )
            mode = "experience_exception"
            reason = "experience_outside_observed_dev_range"
        row.update(
            {
                "scenario_id": scenario_id,
                "source": str(row.get("source") or "manual"),
                "job_category": category,
                "job_title": title,
                "years_of_experience": years,
                "validation_mode": mode,
                "exception_reason": reason,
                "policy_hash": policy["policy_hash"],
            }
        )
        validated.append(row)
    return validated
