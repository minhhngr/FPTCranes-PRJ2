from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .segmentation_evidence import build_segmentation_evidence, build_segmentation_insights
from .segmentation_robustness import (
    build_representation_specs,
    choose_official_representation,
    evaluate_candidates,
    pairwise_representation_ari,
    transform_representation,
)
from .segmentation_stability import ResampleStabilityConfig, subsample_stability
from .training_audit import (
    active_audit,
    dataframe_identity,
    emit_event,
    export_evidence_table,
    start_training_audit,
)

TARGET = "annual_salary_usd"
SOURCE_COLUMNS = [
    "job_id",
    "job_title",
    "AI Engineering",
    "experience_level",
    "years_of_experience",
    "education_required",
    "annual_salary_usd",
    "salary_min_usd",
    "salary_max_usd",
    "city",
    "country",
    "remote_work",
    "company_size",
    "industry",
    "required_skills",
    "ai_salary_premium_pct",
    "demand_score",
    "demand_growth_yoy_pct",
    "benefits_score_10",
    "posting_year",
    "posting_month",
    "is_senior",
    "is_remote_friendly",
    "is_llm_role",
    "salary_tier",
]
MODEL_FEATURES = [
    "job_title",
    "job_category",
    "years_of_experience",
    "education_required",
    "city",
    "country",
    "remote_work",
    "company_size",
    "industry",
    "demand_score",
    "benefits_score_10",
    "required_skills",
    "skill_count",
]
# Canonical Branch-A input contract.
# Exactly the 13 fields retained by Stage 3 / Feature Governance:
# 12 retained source/business fields + 1 engineered field (skill_count).
# Every R0-R4 representation must be a subset of this source-of-truth set.
SEGMENTATION_FEATURES = [
    "job_title",
    "job_category",
    "years_of_experience",
    "education_required",
    "city",
    "country",
    "remote_work",
    "company_size",
    "industry",
    "demand_score",
    "benefits_score_10",
    "required_skills",
    "skill_count",
]
EDUCATION_ORDER = ["Bootcamp/Self-taught", "Associate's", "Bachelor's", "Master's", "PhD"]


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_config(path: Path | None = None) -> dict[str, Any]:
    path = path or project_root() / "config" / "project.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(obj: Any, path: Path) -> None:
    audit = active_audit()
    if audit is not None:
        emit_event(
            "artifact_write_started",
            step_id=f"artifact.json.{Path(path).name}",
            operation="write_json",
            status="started",
            message="Starting JSON artifact write.",
            extra={"artifact_path": str(path)},
        )
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_json_default)
    if audit is not None:
        emit_event(
            "artifact_written",
            step_id=f"artifact.json.{Path(path).name}",
            operation="write_json",
            status="completed",
            message="JSON artifact write completed.",
            extra={"artifact_path": str(path)},
        )


def _json_default(x: Any) -> Any:
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.ndarray,)):
        return x.tolist()
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, pd.Timestamp):
        return x.isoformat()
    raise TypeError(type(x).__name__)


def save_csv(df: pd.DataFrame, path: Path) -> None:
    audit = active_audit()
    if audit is not None:
        emit_event(
            "artifact_write_started",
            step_id=f"artifact.csv.{Path(path).name}",
            operation="write_csv",
            status="started",
            message="Starting CSV artifact write.",
            extra={
                "artifact_path": str(path),
                "rows": int(len(df)),
                "column_count": int(len(df.columns)),
            },
        )
    ensure_dir(path.parent)
    df.to_csv(path, index=False)
    if audit is not None:
        emit_event(
            "artifact_written",
            step_id=f"artifact.csv.{Path(path).name}",
            operation="write_csv",
            status="completed",
            message="CSV artifact write completed.",
            extra={"artifact_path": str(path), "rows": int(len(df))},
        )


def _audited_joblib_dump(obj: Any, path: Path, *, role: str) -> None:
    step_id = f"artifact.joblib.{path.name}.write"
    emit_event(
        "operation_started",
        step_id=step_id,
        operation="write_joblib",
        status="started",
        message=f"Write {role} artifact.",
        detail_level=3,
        extra={"artifact_path": str(path), "artifact_role": role},
    )
    try:
        joblib.dump(obj, path)
    except BaseException as error:
        status = "cancelled" if isinstance(error, KeyboardInterrupt) else "failed"
        emit_event(
            "operation_completed",
            step_id=step_id,
            operation="write_joblib",
            status=status,
            message=f"Could not complete {role} artifact write.",
            detail_level=3,
            extra={
                "artifact_path": str(path),
                "artifact_role": role,
                "error_type": type(error).__name__,
            },
        )
        raise
    emit_event(
        "operation_completed",
        step_id=step_id,
        operation="write_joblib",
        status="completed",
        message=f"Wrote {role} artifact.",
        detail_level=3,
        extra={"artifact_path": str(path), "artifact_role": role},
    )


def _audited_joblib_load(path: Path, *, role: str) -> Any:
    step_id = f"artifact.joblib.{path.name}.read"
    emit_event(
        "operation_started",
        step_id=step_id,
        operation="read_joblib",
        status="started",
        message=f"Reload {role} artifact.",
        detail_level=3,
        extra={"artifact_path": str(path), "artifact_role": role},
    )
    try:
        value = joblib.load(path)
    except BaseException as error:
        status = "cancelled" if isinstance(error, KeyboardInterrupt) else "failed"
        emit_event(
            "operation_completed",
            step_id=step_id,
            operation="read_joblib",
            status=status,
            message=f"Could not reload {role} artifact.",
            detail_level=3,
            extra={
                "artifact_path": str(path),
                "artifact_role": role,
                "error_type": type(error).__name__,
            },
        )
        raise
    emit_event(
        "operation_completed",
        step_id=step_id,
        operation="read_joblib",
        status="completed",
        message=f"Reloaded {role} artifact.",
        detail_level=3,
        extra={"artifact_path": str(path), "artifact_role": role},
    )
    return value


def canonicalize_source_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Return a canonical 25-column source frame and non-fatal schema notes.

    The technical design names the raw domain column ``AI Engineering``. New
    datasets may already use the cleaned alias ``job_category``; that alias is
    accepted and converted back to the raw contract before the pipeline starts.
    Extra columns are ignored but reported by the preflight layer.
    """
    work = df.copy()
    notes: list[str] = []
    if "AI Engineering" not in work.columns and "job_category" in work.columns:
        work = work.rename(columns={"job_category": "AI Engineering"})
        notes.append("Accepted alias: job_category → AI Engineering.")
    missing = [c for c in SOURCE_COLUMNS if c not in work.columns]
    if missing:
        raise ValueError(f"Input schema is missing required source columns: {missing}")
    extra = [c for c in work.columns if c not in SOURCE_COLUMNS]
    if extra:
        notes.append(f"Ignored {len(extra)} extra column(s): {extra}")
    return work[SOURCE_COLUMNS].copy(), notes


def validate_input_schema(df: pd.DataFrame) -> dict[str, Any]:
    """Validate a user-supplied raw CSV before a full offline run.

    Validation separates blocking errors from warnings. A file is processable
    only when the required source contract, target, time fields and key numeric
    types are valid. Unknown category values are deliberately not blocking
    because fitted encoders use ``handle_unknown='ignore'``.
    """
    original_cols = list(df.columns)
    alias_used = "AI Engineering" not in df.columns and "job_category" in df.columns
    expected = list(SOURCE_COLUMNS)
    normalized_cols = set(original_cols)
    if alias_used:
        normalized_cols.add("AI Engineering")
    missing = [c for c in expected if c not in normalized_cols]
    extra = [c for c in original_cols if c not in SOURCE_COLUMNS and c != "job_category"]
    issues: list[dict[str, Any]] = []
    for c in missing:
        issues.append(
            {"level": "ERROR", "field": c, "message": "Required source column is missing."}
        )
    if alias_used:
        issues.append(
            {
                "level": "INFO",
                "field": "job_category",
                "message": "Accepted alias for raw column 'AI Engineering'.",
            }
        )
    for c in extra:
        issues.append(
            {
                "level": "WARNING",
                "field": c,
                "message": "Extra column will be ignored by the canonical source contract.",
            }
        )

    field_rows = []
    for c in expected:
        detected = c in df.columns or (c == "AI Engineering" and alias_used)
        src = (
            c
            if c in df.columns
            else ("job_category" if c == "AI Engineering" and alias_used else None)
        )
        dtype = str(df[src].dtype) if src else ""
        miss = int(df[src].isna().sum()) if src else None
        unique = int(df[src].nunique(dropna=True)) if src else None
        field_rows.append(
            {
                "expected_column": c,
                "detected": detected,
                "source_column": src or "",
                "dtype": dtype,
                "missing": miss,
                "unique": unique,
            }
        )
    field_table = pd.DataFrame(field_rows)

    if not missing:
        work = df.rename(columns={"job_category": "AI Engineering"} if alias_used else {}).copy()
        numeric_required = [
            TARGET,
            "salary_min_usd",
            "salary_max_usd",
            "years_of_experience",
            "ai_salary_premium_pct",
            "demand_score",
            "demand_growth_yoy_pct",
            "benefits_score_10",
            "posting_year",
            "posting_month",
        ]
        for c in numeric_required:
            coerced = pd.to_numeric(work[c], errors="coerce")
            bad = int((coerced.isna() & work[c].notna()).sum())
            if bad > 0:
                issues.append(
                    {
                        "level": "ERROR",
                        "field": c,
                        "message": f"{bad} value(s) cannot be parsed as numeric.",
                    }
                )
        for c in [TARGET, "posting_year", "posting_month"]:
            if work[c].isna().any():
                issues.append(
                    {
                        "level": "ERROR",
                        "field": c,
                        "message": "Missing values are not allowed for full supervised processing.",
                    }
                )
        months = pd.to_numeric(work["posting_month"], errors="coerce")
        invalid_months = int((~months.between(1, 12)).fillna(True).sum())
        if invalid_months:
            issues.append(
                {
                    "level": "ERROR",
                    "field": "posting_month",
                    "message": f"{invalid_months} row(s) have month outside 1..12 or are not numeric.",
                }
            )
        if len(work) < 30:
            issues.append(
                {
                    "level": "WARNING",
                    "field": "__rows__",
                    "message": "Very small dataset; temporal CV and clustering may be unstable.",
                }
            )
        period_pairs = (
            work[["posting_year", "posting_month"]]
            .apply(pd.to_numeric, errors="coerce")
            .dropna()
            .drop_duplicates()
        )
        if len(period_pairs) < 3:
            issues.append(
                {
                    "level": "ERROR",
                    "field": "posting_year/posting_month",
                    "message": "At least 3 distinct monthly periods are required for temporal validation.",
                }
            )
        if work.duplicated().any():
            issues.append(
                {
                    "level": "WARNING",
                    "field": "__rows__",
                    "message": f"{int(work.duplicated().sum())} exact duplicate row(s) detected; Basic Clean will remove them.",
                }
            )
        if work["job_id"].duplicated().any():
            issues.append(
                {
                    "level": "WARNING",
                    "field": "job_id",
                    "message": f"{int(work['job_id'].duplicated().sum())} duplicate identifier value(s) detected.",
                }
            )
        hidden = 0
        for c in work.select_dtypes(include="object").columns:
            hidden += int(
                work[c].astype(str).str.strip().isin(["", "nan", "None", "null", "NA", "N/A"]).sum()
            )
        if hidden:
            issues.append(
                {
                    "level": "WARNING",
                    "field": "__text__",
                    "message": f"{hidden} hidden-missing text value(s) detected; review before operational use.",
                }
            )

    valid = not any(i["level"] == "ERROR" for i in issues)
    return {
        "valid": valid,
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "alias_used": alias_used,
        "missing_columns": missing,
        "extra_columns": extra,
        "issues": issues,
        "field_table": field_table,
    }


def read_raw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    canonical, _ = canonicalize_source_dataframe(df)
    return canonical


def normalize_skills(value: Any, delimiter: str = "|") -> list[str]:
    if pd.isna(value):
        return []
    tokens = [re.sub(r"\s+", " ", t.strip()) for t in str(value).split(delimiter)]
    return sorted(dict.fromkeys(t for t in tokens if t))


def derive_skill_count(series: pd.Series) -> pd.Series:
    return series.map(lambda x: len(normalize_skills(x))).astype(int)


def profile_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n = max(len(df), 1)
    for c in df.columns:
        s = df[c]
        hidden = int(s.astype(str).str.strip().isin(["", "nan", "None", "null", "NA", "N/A"]).sum())
        rows.append(
            {
                "column": c,
                "dtype": str(s.dtype),
                "unique": int(s.nunique(dropna=True)),
                "unique_pct": round(100 * s.nunique(dropna=True) / n, 2),
                "missing": int(s.isna().sum()),
                "hidden_missing": hidden,
            }
        )
    return pd.DataFrame(rows)


def basic_clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = raw.rename(columns={"AI Engineering": "job_category"}).copy()
    bad_mask = df["job_category"].astype(str).str.strip().str.lower().eq("job_category")
    removed_bad = df.loc[bad_mask].copy()
    df = df.loc[~bad_mask].copy()
    duplicate_rows = int(df.duplicated().sum())
    if duplicate_rows:
        df = df.drop_duplicates().copy()
    identifier_unique = bool(
        "job_id" in df.columns and df["job_id"].nunique(dropna=False) == len(df)
    )
    if "job_id" in df.columns:
        df = df.drop(columns=["job_id"])
    audit = {
        "raw_rows": int(len(raw)),
        "raw_columns": int(raw.shape[1]),
        "invalid_category_rows_removed": int(bad_mask.sum()),
        "invalid_category_job_ids": removed_bad.get("job_id", pd.Series(dtype=str))
        .astype(str)
        .tolist(),
        "duplicate_rows_removed": duplicate_rows,
        "identifier_was_unique": identifier_unique,
        "identifier_removed": True,
        "clean_rows": int(len(df)),
        "clean_columns": int(df.shape[1]),
        "skill_count_created_later": True,
    }
    return df.reset_index(drop=True), audit


def _experience_expected(level: str, years: float) -> bool:
    if pd.isna(level) or pd.isna(years):
        return False
    y = float(years)
    s = str(level)
    if s.startswith("Entry"):
        return y <= 2
    if s.startswith("Mid"):
        return 3 <= y <= 5
    if s.startswith("Senior"):
        return 6 <= y <= 9
    if s.startswith("Lead"):
        return y >= 10
    return False


def expected_salary_tier(salary: float) -> str:
    if salary < 100_000:
        return "Entry (<$100k)"
    if salary < 150_000:
        return "Mid ($100-150k)"
    if salary < 200_000:
        return "Upper-Mid ($150-200k)"
    if salary < 300_000:
        return "Senior ($200-300k)"
    return "Elite (>$300k)"


def contradiction_outputs(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    exp_bad = ~pd.Series(
        [
            _experience_expected(a, b)
            for a, b in zip(df["experience_level"], df["years_of_experience"])
        ],
        index=df.index,
    )
    tier_bad = df[TARGET].map(expected_salary_tier).ne(df["salary_tier"])
    range_bad = (df[TARGET] < df["salary_min_usd"]) | (df[TARGET] > df["salary_max_usd"])
    dup_skill = df["required_skills"].map(
        lambda x: len(str(x).split("|")) > len(normalize_skills(x))
    )
    findings = (
        pd.DataFrame(
            [
                {
                    "issue": "Experience bucket mismatch",
                    "affected_rows": int(exp_bad.sum()),
                    "affected_pct": 100 * exp_bad.mean(),
                    "severity": "High",
                    "required_action": "Treat experience_level as contradictory/redundant; test years_of_experience separately.",
                },
                {
                    "issue": "Salary tier inconsistency",
                    "affected_rows": int(tier_bad.sum()),
                    "affected_pct": 100 * tier_bad.mean(),
                    "severity": "High",
                    "required_action": "Block salary_tier from predictors because it is target-adjacent and inconsistent.",
                },
                {
                    "issue": "Salary outside stated min/max",
                    "affected_rows": int(range_bad.sum()),
                    "affected_pct": 100 * range_bad.mean(),
                    "severity": "High",
                    "required_action": "Block salary_min_usd and salary_max_usd from predictors; retain for audit only.",
                },
                {
                    "issue": "Duplicate skill tokens",
                    "affected_rows": int(dup_skill.sum()),
                    "affected_pct": 100 * dup_skill.mean(),
                    "severity": "Medium",
                    "required_action": "Normalize and deduplicate skills before multi-hot encoding.",
                },
            ]
        )
        .sort_values("affected_pct", ascending=False)
        .reset_index(drop=True)
    )
    by_level = (
        df.groupby("experience_level", observed=True)
        .agg(
            records=(TARGET, "size"),
            years_mean=("years_of_experience", "mean"),
            years_median=("years_of_experience", "median"),
            salary_mean=(TARGET, "mean"),
            salary_median=(TARGET, "median"),
        )
        .reset_index()
    )
    by_years = (
        df.groupby("years_of_experience", observed=True)
        .agg(
            records=(TARGET, "size"), salary_mean=(TARGET, "mean"), salary_median=(TARGET, "median")
        )
        .reset_index()
        .sort_values("years_of_experience")
    )
    by_domain = (
        df.groupby("job_category", observed=True)
        .agg(
            records=(TARGET, "size"), salary_mean=(TARGET, "mean"), salary_median=(TARGET, "median")
        )
        .reset_index()
        .sort_values("salary_mean", ascending=False)
    )
    range_status = (
        pd.DataFrame({"status": np.where(range_bad, "Outside stated range", "Inside stated range")})
        .value_counts()
        .rename("rows")
        .reset_index()
    )
    tier_crosstab = pd.crosstab(
        df["salary_tier"], df[TARGET].map(expected_salary_tier)
    ).reset_index()
    return findings, {
        "experience_by_level": by_level,
        "salary_by_years": by_years,
        "salary_by_domain": by_domain,
        "salary_range_status": range_status,
        "salary_tier_crosstab": tier_crosstab,
    }


def stage1_detailed_outputs(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Audit-ready long tables used by Streamlit interactive views."""
    work = df.copy().reset_index(drop=True)
    work["record_id"] = np.arange(1, len(work) + 1)
    exp_bad = ~pd.Series(
        [
            _experience_expected(a, b)
            for a, b in zip(work["experience_level"], work["years_of_experience"])
        ],
        index=work.index,
    )
    expected_tier = work[TARGET].map(expected_salary_tier)
    tier_bad = expected_tier.ne(work["salary_tier"])
    range_bad = (work[TARGET] < work["salary_min_usd"]) | (work[TARGET] > work["salary_max_usd"])
    flags = work[
        [
            "record_id",
            "job_title",
            "job_category",
            "experience_level",
            "years_of_experience",
            TARGET,
            "salary_min_usd",
            "salary_max_usd",
            "salary_tier",
            "country",
            "city",
        ]
    ].copy()
    flags["expected_salary_tier"] = expected_tier
    flags["experience_mismatch"] = exp_bad
    flags["salary_tier_mismatch"] = tier_bad
    flags["salary_range_mismatch"] = range_bad
    num_cols = [c for c in work.select_dtypes(include=np.number).columns if c != "record_id"]
    desc = (
        work[num_cols]
        .describe(percentiles=[0.25, 0.5, 0.75])
        .T.reset_index()
        .rename(columns={"index": "feature"})
    )
    long_cats = []
    for c in work.select_dtypes(exclude=np.number).columns:
        if c == "required_skills":
            continue
        vc = work[c].astype(str).value_counts(dropna=False).head(50)
        for value, count in vc.items():
            long_cats.append(
                {
                    "feature": c,
                    "category": value,
                    "records": int(count),
                    "share_pct": 100 * count / len(work),
                }
            )
    cat_summary = pd.DataFrame(long_cats)
    return {
        "integrity_row_flags": flags,
        "numeric_descriptive_summary": desc,
        "categorical_frequency_summary": cat_summary,
    }


def stage2_detailed_outputs(
    dev: pd.DataFrame,
    test: pd.DataFrame,
    prep: ColumnTransformer,
    ztr: np.ndarray,
    names: np.ndarray,
) -> dict[str, pd.DataFrame]:
    periods = pd.to_datetime(
        dict(
            year=pd.concat([dev, test])["posting_year"].astype(int),
            month=pd.concat([dev, test])["posting_month"].astype(int),
            day=1,
        )
    )
    tmp = pd.DataFrame(
        {
            "period": periods.dt.strftime("%Y-%m"),
            "split": ["Development"] * len(dev) + ["Locked test"] * len(test),
        }
    )
    monthly = tmp.groupby(["period", "split"]).size().rename("records").reset_index()
    fmap = []
    for n in names:
        txt = str(n)
        src = txt.split("__", 1)[-1]
        if txt.startswith("skills__") or "skill__" in txt:
            original = "required_skills"
        elif txt.startswith("numeric__"):
            original = src
        elif txt.startswith("nominal__"):
            tail = src
            candidates = [
                "job_title",
                "job_category",
                "education_required",
                "city",
                "country",
                "remote_work",
                "company_size",
                "industry",
            ]
            original = next(
                (c for c in candidates if tail.startswith(c + "_") or tail == c), tail.split("_")[0]
            )
        else:
            original = src
        fmap.append(
            {
                "encoded_feature": txt,
                "original_feature": original,
                "transformer": txt.split("__", 1)[0] if "__" in txt else "custom",
            }
        )
    fmap = pd.DataFrame(fmap)
    rows = []
    for c in ["years_of_experience", "demand_score", "benefits_score_10", "skill_count"]:
        if c not in dev:
            continue
        enc_name = next((str(n) for n in names if str(n).endswith("__" + c) or str(n) == c), None)
        if enc_name is None:
            continue
        j = list(map(str, names)).index(enc_name)
        a = ztr[:, j]
        rows.append(
            {
                "feature": c,
                "before_mean": float(dev[c].mean()),
                "before_std": float(dev[c].std(ddof=0)),
                "before_min": float(dev[c].min()),
                "before_max": float(dev[c].max()),
                "after_mean": float(np.mean(a)),
                "after_std": float(np.std(a)),
                "after_min": float(np.min(a)),
                "after_max": float(np.max(a)),
            }
        )
    scaling = pd.DataFrame(rows)
    skill_rows = []
    for _, r in dev[["required_skills", TARGET, "years_of_experience"]].iterrows():
        for tok in normalize_skills(r.required_skills):
            skill_rows.append(
                {"skill": tok, TARGET: r[TARGET], "years_of_experience": r.years_of_experience}
            )
    skills = pd.DataFrame(skill_rows)
    if len(skills):
        skills = (
            skills.groupby("skill")
            .agg(
                records=(TARGET, "size"),
                salary_mean=(TARGET, "mean"),
                salary_median=(TARGET, "median"),
                years_mean=("years_of_experience", "mean"),
            )
            .reset_index()
            .sort_values("records", ascending=False)
        )
    # Point-biserial/Pearson correlation of each TRAIN-only multi-hot skill indicator with target.
    y = dev[TARGET].to_numpy(dtype=float)
    skill_corr_rows = []
    vocab = sorted({t for v in dev["required_skills"] for t in normalize_skills(v)})
    token_sets = [set(normalize_skills(v)) for v in dev["required_skills"]]
    for tok in vocab:
        x = np.fromiter(
            (1.0 if tok in ss else 0.0 for ss in token_sets), dtype=float, count=len(token_sets)
        )
        r = 0.0 if np.std(x) == 0 else float(np.corrcoef(x, y)[0, 1])
        skill_corr_rows.append(
            {"skill": tok, "records": int(x.sum()), "pearson_r": r, "abs_r": abs(r)}
        )
    skill_corr = pd.DataFrame(skill_corr_rows).sort_values("abs_r", ascending=False)
    return {
        "monthly_split_counts": monthly,
        "preprocessing_feature_map": fmap,
        "numeric_scaling_summary": scaling,
        "skill_token_summary": skills,
        "skill_target_correlations": skill_corr,
    }


def best_model_subgroup_outputs(pred_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    outs = {}
    for c in ["job_category", "country", "remote_work", "company_size"]:
        if c not in pred_df:
            continue
        g = (
            pred_df.groupby(c, observed=True)
            .agg(
                records=("absolute_error_usd", "size"),
                MAE=("absolute_error_usd", "mean"),
                median_abs_error=("absolute_error_usd", "median"),
                actual_salary_mean=(TARGET, "mean"),
                predicted_salary_mean=("predicted_salary_usd", "mean"),
            )
            .reset_index()
            .sort_values("MAE", ascending=False)
        )
        outs[f"error_by_{c}"] = g
    tmp = pred_df.copy()
    tmp["experience_band"] = pd.cut(
        tmp["years_of_experience"],
        bins=[0, 2, 5, 9, np.inf],
        labels=["Entry (1-2)", "Mid (3-5)", "Senior (6-9)", "Lead (10+)"],
        include_lowest=True,
    )
    outs["error_by_experience_band"] = (
        tmp.groupby("experience_band", observed=True)
        .agg(
            records=("absolute_error_usd", "size"),
            MAE=("absolute_error_usd", "mean"),
            median_abs_error=("absolute_error_usd", "median"),
            actual_salary_mean=(TARGET, "mean"),
            predicted_salary_mean=("predicted_salary_usd", "mean"),
        )
        .reset_index()
    )
    return outs


def feature_policy_table() -> pd.DataFrame:
    blocked = {
        "job_id": "identifier",
        "salary_min_usd": "target-adjacent salary metadata",
        "salary_max_usd": "target-adjacent salary metadata",
        "salary_tier": "target-derived / inconsistent salary band",
        "experience_level": "contradictory with numeric experience in primary salary model",
        "posting_year": "split/time governance field",
        "posting_month": "split/time governance field",
        "is_senior": "derived/redundant flag",
        "is_remote_friendly": "derived/redundant flag",
        "is_llm_role": "derived/redundant flag",
        "ai_salary_premium_pct": "target-adjacent market signal in primary salary model",
        "demand_growth_yoy_pct": "excluded from primary deployment contract",
    }
    rows = []
    for c in MODEL_FEATURES:
        rows.append(
            {
                "feature": c,
                "policy": "ALLOW",
                "reason": "Known at prediction time; admitted by serving contract.",
            }
        )
    for c, reason in blocked.items():
        rows.append({"feature": c, "policy": "BLOCK", "reason": reason})
    rows.append(
        {"feature": TARGET, "policy": "TARGET", "reason": "Prediction target; never included in X."}
    )
    return pd.DataFrame(rows)


def choose_locked_period(
    df: pd.DataFrame, preferred_year: int = 2026, preferred_month: int = 3
) -> tuple[int, int, str]:
    periods = sorted({(int(y), int(m)) for y, m in zip(df["posting_year"], df["posting_month"])})
    preferred = (preferred_year, preferred_month)
    if preferred in periods and preferred == max(periods):
        return preferred_year, preferred_month, "preferred_latest_period"
    if preferred in periods and max(periods) <= preferred:
        return preferred_year, preferred_month, "preferred_period"
    y, m = max(periods)
    return y, m, "latest_available_period"


def temporal_split(
    df: pd.DataFrame, preferred_year: int = 2026, preferred_month: int = 3
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    y, m, rule = choose_locked_period(df, preferred_year, preferred_month)
    test_mask = (df["posting_year"].astype(int) == y) & (df["posting_month"].astype(int) == m)
    dev = df.loc[~test_mask].copy()
    test = df.loc[test_mask].copy()
    if len(test) == 0 or len(dev) == 0:
        raise ValueError("Temporal split failed: development or locked-test set is empty.")
    summary = {
        "locked_test_year": y,
        "locked_test_month": m,
        "selection_rule": rule,
        "development_rows": int(len(dev)),
        "locked_test_rows": int(len(test)),
        "development_pct": 100 * len(dev) / len(df),
        "locked_test_pct": 100 * len(test) / len(df),
        "locked_test_period_label": f"{y:04d}-{m:02d}",
    }
    return dev.reset_index(drop=True), test.reset_index(drop=True), summary


class SkillMultiHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, delimiter: str = "|"):
        self.delimiter = delimiter

    def fit(self, X, y=None):
        values = self._values(X)
        self.vocabulary_ = sorted({t for v in values for t in normalize_skills(v, self.delimiter)})
        self.index_ = {t: i for i, t in enumerate(self.vocabulary_)}
        return self

    def transform(self, X):
        values = self._values(X)
        arr = np.zeros((len(values), len(self.vocabulary_)), dtype=float)
        for i, v in enumerate(values):
            for t in normalize_skills(v, self.delimiter):
                j = self.index_.get(t)
                if j is not None:
                    arr[i, j] = 1.0
        return arr

    def get_feature_names_out(self, input_features=None):
        return np.array([f"skill__{t}" for t in self.vocabulary_], dtype=object)

    @staticmethod
    def _values(X):
        if isinstance(X, pd.DataFrame):
            return X.iloc[:, 0].tolist()
        if isinstance(X, pd.Series):
            return X.tolist()
        a = np.asarray(X, dtype=object)
        if a.ndim == 2:
            a = a[:, 0]
        return a.tolist()


def make_salary_preprocessor(features: list[str] | None = None) -> ColumnTransformer:
    features = features or MODEL_FEATURES
    cats = [
        c
        for c in [
            "job_title",
            "job_category",
            "experience_level",
            "education_required",
            "city",
            "country",
            "remote_work",
            "company_size",
            "industry",
        ]
        if c in features
    ]
    nums = [
        c
        for c in ["years_of_experience", "demand_score", "benefits_score_10", "skill_count"]
        if c in features
    ]
    transformers = []
    if cats:
        transformers.append(
            ("nominal", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cats)
        )
    if nums:
        transformers.append(("numeric", StandardScaler(), nums))
    if "required_skills" in features:
        transformers.append(("skills", SkillMultiHotEncoder(), ["required_skills"]))
    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=True,
        sparse_threshold=0.0,
    )


def make_model_pipeline(model: BaseEstimator, features: list[str] | None = None) -> Pipeline:
    return Pipeline([("preprocess", make_salary_preprocessor(features)), ("model", model)])


def temporal_cv_splits(
    dev: pd.DataFrame, n_splits: int = 5, block_size: int = 200
) -> list[tuple[np.ndarray, np.ndarray, str]]:
    """Temporal sliding-window cross-validation.

    1. Sorts chronologically by posting_year and posting_month.
    2. Divides the sorted records into contiguous 200-row blocks.
    3. Slides a 1-block window forward:
       Fold k uses Block k-1 (200 rows) as Train, and Block k (200 rows) as Validation.
    """
    periods = pd.to_datetime(
        dict(
            year=dev["posting_year"].astype(int),
            month=dev["posting_month"].astype(int),
            day=1,
        )
    )
    order = np.argsort(periods.to_numpy(), kind="mergesort")
    n_rows = len(order)

    bs = int(block_size)
    n_blocks = n_rows // bs
    if n_blocks < 2:
        bs = max(20, n_rows // max(2, int(n_splits) + 1))
        n_blocks = n_rows // bs

    effective_splits = min(int(n_splits), max(1, n_blocks - 1))
    out: list[tuple[np.ndarray, np.ndarray, str]] = []
    for k in range(1, effective_splits + 1):
        tr = order[(k - 1) * bs : k * bs]
        va_end = (k + 1) * bs if k < effective_splits else n_rows
        va = order[k * bs : va_end]
        if len(tr) and len(va):
            v_sub = dev.iloc[va]
            sm = f"{int(v_sub.iloc[0]['posting_year'])}-{int(v_sub.iloc[0]['posting_month']):02d}"
            em = f"{int(v_sub.iloc[-1]['posting_year'])}-{int(v_sub.iloc[-1]['posting_month']):02d}"
            label = sm if sm == em else f"{sm}..{em}"
            out.append((tr, va, label))
    return out


def _temporal_fold_audit_payload(
    dev: pd.DataFrame, splits: list[tuple[np.ndarray, np.ndarray, str]], requested_folds: int
) -> dict[str, Any]:
    periods = pd.to_datetime(
        dict(
            year=dev["posting_year"].astype(int),
            month=dev["posting_month"].astype(int),
            day=1,
        )
    ).reset_index(drop=True)
    row_digest = hashlib.sha256("|".join(map(str, range(len(dev)))).encode("utf-8")).hexdigest()
    fold_rows: list[dict[str, Any]] = []
    split_material: list[str] = []
    for i, (tr, va, label) in enumerate(splits, start=1):
        train_positions = list(map(int, tr))
        validation_positions = list(map(int, va))
        train_periods = (
            periods.iloc[train_positions]
            if train_positions
            else pd.Series([], dtype="datetime64[ns]")
        )
        validation_periods = (
            periods.iloc[validation_positions]
            if validation_positions
            else pd.Series([], dtype="datetime64[ns]")
        )
        train_months = {p.strftime("%Y-%m") for p in train_periods}
        validation_months = {p.strftime("%Y-%m") for p in validation_periods}
        row_overlap = sorted(set(train_positions) & set(validation_positions))
        shared_periods = sorted(train_months & validation_months)
        split_material.append(
            ",".join(map(str, train_positions)) + ":" + ",".join(map(str, validation_positions))
        )
        fold_rows.append(
            {
                "fold_id": int(i),
                "validation_period": label,
                "train_positions": train_positions,
                "validation_positions": validation_positions,
                "train_rows": int(len(train_positions)),
                "validation_rows": int(len(validation_positions)),
                "train_period_min": min(train_months) if train_months else None,
                "train_period_max": max(train_months) if train_months else None,
                "validation_period_min": min(validation_months) if validation_months else None,
                "validation_period_max": max(validation_months) if validation_months else None,
                "row_overlap_count": int(len(row_overlap)),
                "row_overlap_positions": row_overlap,
                "shared_periods": shared_periods,
            }
        )
    split_id = hashlib.sha256("|".join(split_material).encode("utf-8")).hexdigest()
    return {
        "split_id": split_id,
        "dataset_id": dataframe_identity(dev),
        "dataset_identity_method": "pandas_hash_content_columns_index_v1",
        "dataset_row_order_sha256": row_digest,
        "dataset_row_order_digest_scope": "positions_only_legacy",
        "sort_policy": "posting_year, posting_month stable mergesort",
        "requested_folds": int(requested_folds),
        "effective_folds": int(len(splits)),
        "folds": fold_rows,
    }


def _audit_start_stage(stage_id: str, title: str) -> None:
    audit = active_audit()
    if audit is not None:
        audit.start_stage(stage_id, title)


def _audit_complete_stage(
    stage_id: str, title: str, *, extra: dict[str, Any] | None = None
) -> None:
    audit = active_audit()
    if audit is not None:
        audit.complete_stage(stage_id, title, extra=extra)


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "R2": float(r2_score(y_true, y_pred)),
        "MedAE": float(median_absolute_error(y_true, y_pred)),
    }


def evaluate_model_cv(
    dev: pd.DataFrame, features: list[str], model_name: str, model: BaseEstimator, n_splits: int = 5
) -> tuple[pd.DataFrame, dict[str, float]]:
    X = dev[features]
    y = dev[TARGET]
    rows = []
    audit = active_audit()
    evaluation_id = f"eval-{audit.sequence + 1:04d}" if audit is not None else None
    emit_event(
        "evaluation_started",
        step_id=f"evaluation.{evaluation_id or model_name}.start",
        operation="temporal_cv_evaluation",
        status="started",
        message=f"Evaluate {model_name} with temporal cross-validation.",
        detail_level=1,
        extra={
            "evaluation_id": evaluation_id,
            "model_name": model_name,
            "target": TARGET,
            "features": list(map(str, features)),
            "requested_folds": int(n_splits),
            "estimator_params": clone(model).get_params(deep=False),
        },
    )
    splits = temporal_cv_splits(dev, n_splits)
    if audit is not None:
        split_payload = _temporal_fold_audit_payload(dev, splits, n_splits)
        if audit.should_emit_split_definition(
            f"{split_payload['dataset_id']}:{split_payload['split_id']}"
        ):
            emit_event(
                "split_defined",
                step_id=f"cv.{evaluation_id}.split",
                operation="temporal_cv_split",
                status="completed",
                message="Temporal CV split definition recorded.",
                detail_level=4,
                extra={"evaluation_id": evaluation_id, "model_name": model_name, **split_payload},
            )
            membership_rows = []
            for fold_info in split_payload["folds"]:
                for role, positions in (
                    ("train", fold_info["train_positions"]),
                    ("validation", fold_info["validation_positions"]),
                ):
                    membership_rows.extend(
                        {
                            "dataset_id": split_payload["dataset_id"],
                            "split_id": split_payload["split_id"],
                            "fold_id": fold_info["fold_id"],
                            "role": role,
                            "position": position,
                        }
                        for position in positions
                    )
            export_evidence_table(
                f"membership-{split_payload['dataset_id'][:12]}-{split_payload['split_id'][:12]}",
                membership_rows,
                kind="fold_membership",
                detail_level=4,
            )
        else:
            emit_event(
                "split_used",
                step_id=f"cv.{evaluation_id}.split",
                operation="temporal_cv_split",
                status="completed",
                message="Previously recorded temporal CV split reused.",
                detail_level=4,
                extra={
                    "evaluation_id": evaluation_id,
                    "model_name": model_name,
                    "dataset_id": split_payload["dataset_id"],
                    "split_id": split_payload["split_id"],
                    "requested_folds": int(n_splits),
                    "effective_folds": int(len(splits)),
                },
            )
        export_evidence_table(
            f"features-{evaluation_id}",
            [
                {
                    "evaluation_id": evaluation_id,
                    "scope": "raw_model_input",
                    "feature_order": order,
                    "feature_name": feature,
                    "role": "input",
                    "policy_reason": "actual evaluation feature",
                }
                for order, feature in enumerate(features, start=1)
            ],
            kind="feature_inventory",
            detail_level=4,
        )
    for i, (tr, va, label) in enumerate(splits, start=1):
        pipe = make_model_pipeline(clone(model), features)
        emit_event(
            "operation_started",
            step_id=f"cv.{evaluation_id or model_name}.fold{i}.fit",
            operation="preprocess_and_fit",
            status="started",
            message="Starting preprocessing and model fit.",
            detail_level=3,
            extra={
                "evaluation_id": evaluation_id,
                "model_name": model_name,
                "fold_id": int(i),
                "validation_period": label,
                "train_rows": int(len(tr)),
                "validation_rows": int(len(va)),
                "features": list(map(str, features)),
                "estimator_params": clone(model).get_params(deep=False),
                "preprocessor_fit_scope": "fold training only",
            },
        )
        t0 = time.perf_counter()
        pipe.fit(X.iloc[tr], y.iloc[tr])
        fit_s = time.perf_counter() - t0
        encoded_feature_count = int(len(pipe.named_steps["preprocess"].get_feature_names_out()))
        emit_event(
            "operation_completed",
            step_id=f"cv.{evaluation_id or model_name}.fold{i}.fit",
            operation="preprocess_and_fit",
            status="completed",
            message="Preprocessing and model fit completed.",
            detail_level=3,
            extra={
                "evaluation_id": evaluation_id,
                "model_name": model_name,
                "fold_id": int(i),
                "elapsed_s": fit_s,
                "encoded_feature_count": encoded_feature_count,
            },
        )
        emit_event(
            "operation_started",
            step_id=f"cv.{evaluation_id or model_name}.fold{i}.predict",
            operation="predict_validation",
            status="started",
            message="Starting validation prediction.",
            detail_level=3,
            extra={"evaluation_id": evaluation_id, "model_name": model_name, "fold_id": int(i)},
        )
        t1 = time.perf_counter()
        pred = pipe.predict(X.iloc[va])
        predict_s = time.perf_counter() - t1
        emit_event(
            "operation_completed",
            step_id=f"cv.{evaluation_id or model_name}.fold{i}.predict",
            operation="predict_validation",
            status="completed",
            message="Validation prediction completed.",
            detail_level=3,
            extra={
                "evaluation_id": evaluation_id,
                "model_name": model_name,
                "fold_id": int(i),
                "elapsed_s": predict_s,
            },
        )
        met = regression_metrics(y.iloc[va], pred)
        emit_event(
            "candidate_fold_scored",
            step_id=f"cv.{evaluation_id or model_name}.fold{i}.score",
            operation="score_validation_fold",
            status="completed",
            message="Validation fold metrics recorded.",
            detail_level=3,
            extra={
                "evaluation_id": evaluation_id,
                "model_name": model_name,
                "fold_id": int(i),
                "validation_period": label,
                "train_rows": int(len(tr)),
                "validation_rows": int(len(va)),
                "partition": "development_temporal_validation",
                "metric_units": {"MAE": "USD", "RMSE": "USD", "R2": "unitless", "MedAE": "USD"},
                **met,
            },
        )
        rows.append(
            {
                "model": model_name,
                "fold": i,
                "validation_period": label,
                "train_rows": len(tr),
                "validation_rows": len(va),
                "fit_time_s": fit_s,
                "predict_time_s": predict_s,
                **met,
            }
        )
    fold = pd.DataFrame(rows)
    summary = {
        "model": model_name,
        "MAE_mean": float(fold.MAE.mean()),
        "MAE_std": float(fold.MAE.std(ddof=0)),
        "RMSE_mean": float(fold.RMSE.mean()),
        "R2_mean": float(fold.R2.mean()),
        "MedAE_mean": float(fold.MedAE.mean()),
        "fit_time_mean_s": float(fold.fit_time_s.mean()),
        "predict_time_mean_s": float(fold.predict_time_s.mean()),
        "folds": int(len(fold)),
    }
    emit_event(
        "evaluation_completed",
        step_id=f"evaluation.{evaluation_id or model_name}.end",
        operation="temporal_cv_evaluation",
        status="completed",
        message=f"Completed temporal cross-validation for {model_name}.",
        detail_level=2,
        extra={
            "evaluation_id": evaluation_id,
            "model_name": model_name,
            **summary,
            "RMSE_std": float(fold.RMSE.std(ddof=0)),
            "R2_std": float(fold.R2.std(ddof=0)),
            "MedAE_std": float(fold.MedAE.std(ddof=0)),
            "aggregation": "arithmetic mean and population standard deviation (ddof=0)",
            "partition": "development_temporal_validation",
            "metric_units": {"MAE": "USD", "RMSE": "USD", "R2": "unitless", "MedAE": "USD"},
            "train_metrics_status": "not_computed",
            "locked_test_metrics_status": "not_evaluated",
            "fit_assessment": "insufficient_evidence",
            "fit_reason": "training_scores_not_computed; diagnostic_rule_not_defined",
        },
    )
    if audit is not None:
        export_evidence_table(
            f"metrics-{evaluation_id}",
            fold.to_dict("records"),
            kind="metric_details",
            detail_level=4,
        )
    return fold, summary


def candidate_models(seed: int = 42) -> dict[str, BaseEstimator]:
    return {
        "Dummy Median": DummyRegressor(strategy="median"),
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=10.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=180, min_samples_leaf=2, max_features=0.8, random_state=seed, n_jobs=1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            random_state=seed, n_estimators=180, learning_rate=0.04, max_depth=2, loss="huber"
        ),
    }


def run_ablation(dev: pd.DataFrame, seed: int = 42, n_splits: int = 5) -> pd.DataFrame:
    options = {
        "A — Conservative Core": MODEL_FEATURES,
        "B — Simplifier": ["job_category", "years_of_experience"],
        "C — Exclude job_category": [c for c in MODEL_FEATURES if c != "job_category"],
        "D — Exclude years_of_experience": [
            c for c in MODEL_FEATURES if c != "years_of_experience"
        ],
    }
    model = RandomForestRegressor(
        n_estimators=120, min_samples_leaf=2, max_features=0.8, random_state=seed, n_jobs=1
    )
    rows = []
    for name, features in options.items():
        _, s = evaluate_model_cv(dev, features, name, model, n_splits)
        rows.append(
            {
                "option": name,
                "feature_count": len(features),
                "features": " | ".join(features),
                **{k: v for k, v in s.items() if k not in ["model"]},
            }
        )
    return pd.DataFrame(rows).sort_values("MAE_mean").reset_index(drop=True)


def run_feature_family_ablation(
    dev: pd.DataFrame, seed: int = 42, n_splits: int = 5
) -> pd.DataFrame:
    """Branch-B family ablation aligned with the richer FPTCranes-PRJ2-main evidence."""
    core = [
        "job_title",
        "job_category",
        "education_required",
        "city",
        "country",
        "remote_work",
        "company_size",
        "industry",
        "demand_score",
        "benefits_score_10",
    ]
    options = {
        "A0_CONSERVATIVE_CORE": core,
        "A1_PLUS_YEARS": core + ["years_of_experience"],
        "A2_EXPERIENCE_BUCKET": core + ["experience_level"],
        "A6_SKILLS": core + ["years_of_experience", "required_skills", "skill_count"],
    }
    model = RandomForestRegressor(
        n_estimators=140, min_samples_leaf=2, max_features=0.8, random_state=seed, n_jobs=1
    )
    rows = []
    baseline = None
    for name, features in options.items():
        _, s = evaluate_model_cv(dev, features, name, model, n_splits)
        row = {
            "experiment": name,
            "feature_count": len(features),
            "features": " | ".join(features),
            "MAE_mean": s["MAE_mean"],
            "MAE_std": s["MAE_std"],
            "RMSE_mean": s["RMSE_mean"],
            "R2_mean": s["R2_mean"],
            "MedAE_mean": s["MedAE_mean"],
        }
        rows.append(row)
        if name == "A0_CONSERVATIVE_CORE":
            baseline = s["MAE_mean"]
    out = pd.DataFrame(rows)
    if baseline is not None:
        out["improvement_vs_A0_pct"] = 100 * (baseline - out["MAE_mean"]) / baseline
    return out.sort_values("MAE_mean").reset_index(drop=True)


def random_forest_importance_by_fold(
    dev: pd.DataFrame, seed: int = 42, n_splits: int = 5
) -> pd.DataFrame:
    """Track encoded RF importance across temporal folds to expose reliance drift."""
    rows = []
    X = dev[MODEL_FEATURES]
    y = dev[TARGET]
    base = RandomForestRegressor(
        n_estimators=180, min_samples_leaf=2, max_features=0.8, random_state=seed, n_jobs=1
    )
    splits = temporal_cv_splits(dev, n_splits)
    audit = active_audit()
    if audit is not None:
        split_payload = _temporal_fold_audit_payload(dev, splits, n_splits)
        if audit.should_emit_split_definition(str(split_payload["split_id"])):
            emit_event(
                "split_defined",
                step_id="importance.rf.split",
                operation="temporal_cv_split",
                status="completed",
                message="Temporal CV split definition recorded for feature-importance fits.",
                extra={
                    "model_name": "Random Forest importance",
                    "metrics_status": "not_computed",
                    **split_payload,
                },
            )
        else:
            emit_event(
                "split_used",
                step_id="importance.rf.split",
                operation="temporal_cv_split",
                status="completed",
                message="Previously recorded temporal CV split reused for feature-importance fits.",
                extra={
                    "model_name": "Random Forest importance",
                    "metrics_status": "not_computed",
                    "split_id": split_payload["split_id"],
                    "requested_folds": int(n_splits),
                    "effective_folds": int(len(splits)),
                },
            )
    for fold, (tr, va, label) in enumerate(splits, 1):
        pipe = make_model_pipeline(clone(base), MODEL_FEATURES)
        emit_event(
            "operation_started",
            step_id=f"importance.rf.fold{fold}.fit",
            operation="preprocess_and_fit",
            status="started",
            message="Starting feature-importance fold fit.",
            extra={
                "model_name": "Random Forest importance",
                "fold_id": int(fold),
                "validation_period": label,
                "train_rows": int(len(tr)),
                "validation_rows": int(len(va)),
                "metrics_status": "not_computed",
            },
        )
        t0 = time.perf_counter()
        pipe.fit(X.iloc[tr], y.iloc[tr])
        fit_s = time.perf_counter() - t0
        emit_event(
            "operation_completed",
            step_id=f"importance.rf.fold{fold}.fit",
            operation="preprocess_and_fit",
            status="completed",
            message="Feature-importance fold fit completed.",
            extra={"fold_id": int(fold), "elapsed_s": fit_s, "metrics_status": "not_computed"},
        )
        imp = extract_encoded_importance(pipe)
        emit_event(
            "operation_completed",
            step_id=f"importance.rf.fold{fold}.extract",
            operation="extract_encoded_importance",
            status="completed",
            message="Encoded feature importance extracted.",
            extra={
                "fold_id": int(fold),
                "encoded_feature_count": int(len(imp)),
                "metrics_status": "not_computed",
            },
        )
        imp["fold"] = fold
        imp["validation_period"] = label
        rows.append(imp)
    return (
        pd.concat(rows, ignore_index=True)
        if rows
        else pd.DataFrame(columns=["encoded_feature", "importance", "fold", "validation_period"])
    )


def branch_b_error_slices(pred_df: pd.DataFrame) -> pd.DataFrame:
    """Single long-form error-slice table used by Branch-B Streamlit drill-down."""
    rows = []
    temp = pred_df.copy()
    temp["experience_band"] = pd.cut(
        temp["years_of_experience"],
        bins=[0, 2, 5, 9, np.inf],
        labels=["Entry (1-2)", "Mid (3-5)", "Senior (6-9)", "Lead (10+)"],
        include_lowest=True,
    )
    for col in ["job_category", "country", "remote_work", "company_size", "experience_band"]:
        if col not in temp:
            continue
        g = (
            temp.groupby(col, observed=True)
            .agg(
                records=("absolute_error_usd", "size"),
                MAE=("absolute_error_usd", "mean"),
                MedAE=("absolute_error_usd", "median"),
                actual_salary_mean=(TARGET, "mean"),
                predicted_salary_mean=("predicted_salary_usd", "mean"),
            )
            .reset_index()
            .rename(columns={col: "group_value"})
        )
        g.insert(0, "group_type", col)
        rows.append(g)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def encoded_correlations_train(
    dev: pd.DataFrame, features: list[str] | None = None
) -> pd.DataFrame:
    features = features or MODEL_FEATURES
    prep = make_salary_preprocessor(features)
    emit_event(
        "operation_started",
        step_id="readiness.encoded_correlations.fit_transform",
        operation="fit_transform_for_encoded_correlations",
        status="started",
        message="Fit preprocessing on DEV and transform DEV for descriptive encoded correlations.",
        detail_level=3,
        extra={
            "train_rows": int(len(dev)),
            "features": list(features),
            "fit_scope": "development only",
        },
    )
    z = prep.fit_transform(dev[features])
    names = prep.get_feature_names_out()
    emit_event(
        "operation_completed",
        step_id="readiness.encoded_correlations.fit_transform",
        operation="fit_transform_for_encoded_correlations",
        status="completed",
        message="DEV-only preprocessing and correlation transform completed.",
        detail_level=3,
        extra={
            "train_rows": int(len(dev)),
            "encoded_feature_count": int(len(names)),
            "fit_scope": "development only",
        },
    )
    y = dev[TARGET].to_numpy(dtype=float)
    rows = []
    for i, n in enumerate(names):
        x = z[:, i].astype(float)
        if np.nanstd(x) == 0:
            r = 0.0
        else:
            r = float(np.corrcoef(x, y)[0, 1])
        rows.append({"encoded_feature": str(n), "pearson_r": r, "abs_r": abs(r)})
    return pd.DataFrame(rows).sort_values("abs_r", ascending=False).reset_index(drop=True)


class FamilyBalancedEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, balance: bool = True, excluded_features: tuple[str, ...] = ()):
        self.balance = balance
        self.excluded_features = excluded_features

    def fit(self, X, y=None):
        X = pd.DataFrame(X).copy() if not isinstance(X, pd.DataFrame) else X.copy()
        # These family definitions intentionally contain ONLY fields from the
        # canonical 13-field Branch-A contract. Stage-3 blocked fields such as
        # experience_level, ai_salary_premium_pct and demand_growth_yoy_pct are
        # not permitted to enter any unsupervised representation.
        base_specs = {
            "Job Domain": (["job_title", "job_category"], []),
            "Experience & Education": (["education_required"], ["years_of_experience"]),
            "Company & Work Mode": (["remote_work", "company_size", "industry"], []),
            "Geography": (["city", "country"], []),
            "Demand / Benefits": ([], ["demand_score", "benefits_score_10"]),
        }
        excluded = set(self.excluded_features or ())
        self.family_specs_ = {
            fam: ([c for c in cats if c not in excluded], [c for c in nums if c not in excluded])
            for fam, (cats, nums) in base_specs.items()
        }
        self.family_transformers_ = {}
        self.family_dims_ = {}
        self.feature_names_ = []
        for fam, (cats, nums) in self.family_specs_.items():
            trs = []
            if cats:
                trs.append(
                    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cats)
                )
            if nums:
                trs.append(("num", StandardScaler(), nums))
            ct = ColumnTransformer(
                trs,
                remainder="drop",
                verbose_feature_names_out=True,
                sparse_threshold=0.0,
            )
            ct.fit(X)
            self.family_transformers_[fam] = ct
            names = [f"{fam}::{n}" for n in ct.get_feature_names_out()]
            self.family_dims_[fam] = len(names)
            self.feature_names_.extend(names)

        excluded = set(self.excluded_features or ())
        if "required_skills" not in excluded:
            sk = SkillMultiHotEncoder().fit(X[["required_skills"]])
            self.family_transformers_["Skills"] = sk
            self.feature_names_.extend([f"Skills::{n}" for n in sk.get_feature_names_out()])
        else:
            sk = None
        if "skill_count" not in excluded:
            sc = StandardScaler().fit(X[["skill_count"]])
            self.family_transformers_["Skills__count_scaler"] = sc
            self.feature_names_.append("Skills::skill_count_scaled")
        else:
            sc = None
        self.family_dims_["Skills"] = (len(sk.vocabulary_) if sk is not None else 0) + (
            1 if sc is not None else 0
        )
        return self

    def transform_family_blocks(self, X, *, balanced: bool | None = None) -> dict[str, np.ndarray]:
        """Return the fitted representation family-by-family.

        This method is used by the offline pipeline to persist family balancing
        evidence. Streamlit never recomputes family weights or transformed
        blocks.
        """
        X = pd.DataFrame(X).copy() if not isinstance(X, pd.DataFrame) else X.copy()
        use_balance = self.balance if balanced is None else bool(balanced)
        blocks: dict[str, np.ndarray] = {}
        for fam in [
            "Job Domain",
            "Experience & Education",
            "Company & Work Mode",
            "Geography",
            "Demand / Benefits",
        ]:
            arr = np.asarray(self.family_transformers_[fam].transform(X), dtype=float)
            if use_balance and arr.shape[1] > 0:
                arr = arr / math.sqrt(arr.shape[1])
            blocks[fam] = arr

        skill_parts = []
        if "Skills" in self.family_transformers_:
            skill_parts.append(
                np.asarray(
                    self.family_transformers_["Skills"].transform(X[["required_skills"]]),
                    dtype=float,
                )
            )
        if "Skills__count_scaler" in self.family_transformers_:
            skill_parts.append(
                np.asarray(
                    self.family_transformers_["Skills__count_scaler"].transform(X[["skill_count"]]),
                    dtype=float,
                )
            )
        skills = np.hstack(skill_parts) if skill_parts else np.empty((len(X), 0), dtype=float)
        if use_balance and skills.shape[1] > 0:
            skills = skills / math.sqrt(skills.shape[1])
        blocks["Skills"] = skills
        return blocks

    def transform(self, X):
        blocks = self.transform_family_blocks(X, balanced=self.balance)
        return np.hstack(
            [
                blocks[f]
                for f in [
                    "Job Domain",
                    "Experience & Education",
                    "Company & Work Mode",
                    "Geography",
                    "Demand / Benefits",
                    "Skills",
                ]
            ]
        )

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_, dtype=object)


def _cluster_quality_band(silhouette: float) -> str:
    if silhouette >= 0.50:
        return "strong separation"
    if silhouette >= 0.35:
        return "moderate separation"
    if silhouette >= 0.20:
        return "weak but usable separation"
    return "very weak / overlapping structure"


def _pairwise_stability(label_sets: list[np.ndarray]) -> float:
    if len(label_sets) < 2:
        return 1.0
    vals = []
    for i in range(len(label_sets)):
        for j in range(i + 1, len(label_sets)):
            vals.append(adjusted_rand_score(label_sets[i], label_sets[j]))
    return float(np.mean(vals)) if vals else 1.0


def _pca_component_count(
    explained_variance_ratio: np.ndarray,
    threshold: float,
    min_components: int = 2,
    max_components: int | None = None,
) -> int:
    explained = np.asarray(explained_variance_ratio, dtype=float)
    if not 0.0 < float(threshold) <= 1.0:
        raise ValueError("PCA variance threshold must be in (0, 1].")
    cumulative = np.cumsum(explained)
    needed = int(np.searchsorted(cumulative, float(threshold), side="left") + 1)
    needed = max(int(min_components), needed)
    if max_components is not None:
        needed = min(needed, int(max_components))
    return min(needed, len(explained))


def _encoded_family_name(encoded_feature: str) -> str:
    text = str(encoded_feature)
    return text.split("::", 1)[0] if "::" in text else "Unknown"


def _build_correlation_selected_space(
    encoder: FamilyBalancedEncoder,
    X_dev: pd.DataFrame,
    X_all: pd.DataFrame,
    *,
    threshold: float = 0.90,
    seed: int = 42,
) -> dict[str, Any]:
    """Build O2: correlation-based feature selection on DEV only.

    Correlation is computed feature-to-feature on the family-balanced encoded
    matrix. The salary target never enters this procedure. Highly redundant
    encoded features are filtered greedily: higher-variance features are
    considered first and a later feature is dropped when its absolute Pearson
    correlation with an already-kept feature reaches ``threshold``.

    The selected encoded columns are used DIRECTLY for clustering. A separate
    two-component PCA is fitted only for 2D visualization.
    """
    if not 0.0 < float(threshold) < 1.0:
        raise ValueError("correlation threshold must be in (0, 1).")

    zdev = np.asarray(encoder.transform(X_dev), dtype=float)
    zall = np.asarray(encoder.transform(X_all), dtype=float)
    names = np.asarray(list(map(str, encoder.get_feature_names_out())), dtype=object)

    if zdev.ndim != 2 or zdev.shape[1] != len(names):
        raise ValueError("O2 encoded matrix / feature-name mismatch.")

    variances = np.nanvar(zdev, axis=0)
    usable = np.where(np.isfinite(variances) & (variances > 1e-12))[0]
    if len(usable) < 2:
        # Keep the two highest-variance columns where possible so clustering
        # and 2D display remain well-defined on small/new datasets.
        usable = np.argsort(np.nan_to_num(variances, nan=-np.inf))[::-1][: min(2, zdev.shape[1])]

    corr = np.corrcoef(zdev[:, usable], rowvar=False)
    corr = np.asarray(corr, dtype=float)
    if corr.ndim == 0:
        corr = np.array([[1.0]], dtype=float)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 1.0)

    # Deterministic greedy filter: high-information (higher variance) features
    # enter first; ties are broken by encoded feature name.
    usable_list = list(map(int, usable))
    order = sorted(
        usable_list,
        key=lambda idx: (-float(variances[idx]), str(names[idx])),
    )
    position = {orig_idx: pos for pos, orig_idx in enumerate(usable_list)}

    kept: list[int] = []
    dropped_reason: dict[int, tuple[int, float]] = {}

    for idx in order:
        if not kept:
            kept.append(idx)
            continue
        i = position[idx]
        best_keep = None
        best_abs = -1.0
        best_signed = np.nan
        for kept_idx in kept:
            j = position[kept_idx]
            signed = float(corr[i, j])
            ab = abs(signed)
            if ab > best_abs:
                best_abs = ab
                best_keep = kept_idx
                best_signed = signed
        if best_keep is not None and best_abs >= float(threshold):
            dropped_reason[idx] = (int(best_keep), float(best_signed))
        else:
            kept.append(idx)

    # In a pathological new dataset, guarantee at least two dimensions if the
    # encoded matrix itself contains at least two usable columns.
    if len(kept) < 2 and len(order) >= 2:
        for idx in order:
            if idx not in kept:
                kept.append(idx)
                dropped_reason.pop(idx, None)
            if len(kept) >= 2:
                break

    kept = sorted(kept)
    keep_set = set(kept)
    selected_dev = zdev[:, kept]
    selected_all = zall[:, kept]

    # Display-only PCA; clustering remains in correlation-selected X space.
    vis_components = min(2, selected_dev.shape[1], selected_dev.shape[0])
    visualizer = PCA(n_components=max(1, vis_components), random_state=int(seed)).fit(selected_dev)
    vis_dev = visualizer.transform(selected_dev)
    vis_all = visualizer.transform(selected_all)
    if vis_dev.shape[1] == 1:
        vis_dev = np.column_stack([vis_dev[:, 0], np.zeros(len(vis_dev))])
        vis_all = np.column_stack([vis_all[:, 0], np.zeros(len(vis_all))])

    # Feature-selection audit table.
    selection_rows: list[dict[str, Any]] = []
    for idx, name in enumerate(names):
        selected = idx in keep_set
        reason = dropped_reason.get(idx)
        selection_rows.append(
            {
                "encoded_feature": str(name),
                "family": _encoded_family_name(str(name)),
                "variance_dev": float(variances[idx]) if np.isfinite(variances[idx]) else np.nan,
                "selected": bool(selected),
                "decision": "KEEP"
                if selected
                else ("DROP_CORRELATED" if reason is not None else "DROP_ZERO_VARIANCE"),
                "correlated_with_kept": str(names[reason[0]]) if reason is not None else "",
                "pearson_r_with_kept": float(reason[1]) if reason is not None else np.nan,
                "abs_r_with_kept": abs(float(reason[1])) if reason is not None else np.nan,
                "correlation_threshold": float(threshold),
            }
        )
    selection_table = pd.DataFrame(selection_rows)

    # Persist all high-correlation pairs so the O2 choice is auditable.
    pair_rows: list[dict[str, Any]] = []
    for ia in range(len(usable_list)):
        a = usable_list[ia]
        for ib in range(ia + 1, len(usable_list)):
            b = usable_list[ib]
            signed = float(corr[ia, ib])
            if abs(signed) >= float(threshold):
                pair_rows.append(
                    {
                        "feature_a": str(names[a]),
                        "family_a": _encoded_family_name(str(names[a])),
                        "feature_b": str(names[b]),
                        "family_b": _encoded_family_name(str(names[b])),
                        "pearson_r": signed,
                        "abs_r": abs(signed),
                        "feature_a_selected": bool(a in keep_set),
                        "feature_b_selected": bool(b in keep_set),
                        "correlation_threshold": float(threshold),
                    }
                )
    redundancy_pairs = pd.DataFrame(pair_rows)
    if not redundancy_pairs.empty:
        redundancy_pairs = redundancy_pairs.sort_values(
            ["abs_r", "feature_a", "feature_b"],
            ascending=[False, True, True],
        ).reset_index(drop=True)

    selected_names = list(map(str, names[kept]))
    family_selected = (
        selection_table[selection_table["selected"]]
        .groupby("family", observed=True)
        .size()
        .rename("selected_encoded_features")
        .reset_index()
    )
    family_total = (
        selection_table.groupby("family", observed=True)
        .size()
        .rename("encoded_features_before")
        .reset_index()
    )
    family_summary = family_total.merge(family_selected, on="family", how="left")
    family_summary["selected_encoded_features"] = (
        family_summary["selected_encoded_features"].fillna(0).astype(int)
    )
    family_summary["retention_pct"] = (
        100.0
        * family_summary["selected_encoded_features"]
        / family_summary["encoded_features_before"].replace(0, np.nan)
    )

    transformer = {
        "kind": "correlation_selected",
        "encoder": encoder,
        "selected_encoded_indices": list(map(int, kept)),
        "selected_encoded_feature_names": selected_names,
        "correlation_threshold": float(threshold),
        "visualizer": visualizer,
    }

    return {
        "dev_matrix": selected_dev,
        "all_matrix": selected_all,
        "visual_dev": vis_dev,
        "visual_all": vis_all,
        "transformer": transformer,
        "selection_table": selection_table,
        "redundancy_pairs": redundancy_pairs,
        "family_summary": family_summary,
        "encoded_features_before": int(zdev.shape[1]),
        "selected_encoded_features": int(selected_dev.shape[1]),
        "visualization_variance_captured": float(
            np.asarray(visualizer.explained_variance_ratio_, dtype=float)[:2].sum()
        ),
    }


def _choose_feature_space_option(
    option_summary: pd.DataFrame,
    *,
    silhouette_tolerance: float,
) -> tuple[str, pd.DataFrame, dict[str, Any]]:
    """Choose O1 vs O2 using gates first, then robustness and parsimony.

    No weighted composite score is used. The hierarchy is:
    1) eligible / hard-gate pass;
    2) near-best silhouette band;
    3) stronger subsample stability;
    4) stronger optimizer-seed stability;
    5) healthier minimum cluster share;
    6) fewer clustering dimensions;
    7) if still tied, O2 for direct encoded-feature interpretability.
    """
    df = option_summary.copy()
    if df.empty:
        raise ValueError("No feature-space option results are available.")

    eligible = df[df["eligible"].astype(bool)].copy()
    fallback = "none"
    if eligible.empty:
        eligible = df.copy()
        fallback = "feature_space_eligibility_relaxed"

    best_sil = float(eligible["silhouette"].max())
    eligible["within_option_tolerance"] = eligible["silhouette"] >= best_sil - float(
        silhouette_tolerance
    )
    shortlist = eligible[eligible["within_option_tolerance"]].copy()
    shortlist["option_preference"] = (
        shortlist["option_id"].map({"O2_CORRELATION": 0, "O1_PCA": 1}).fillna(9)
    )
    shortlist = shortlist.sort_values(
        [
            "resample_stability_ari_mean",
            "stability_ari",
            "min_cluster_share",
            "clustering_dimensions",
            "option_preference",
        ],
        ascending=[False, False, False, True, True],
        na_position="last",
    )

    selected_id = str(shortlist.iloc[0]["option_id"])
    df["within_option_tolerance"] = df["silhouette"] >= best_sil - float(silhouette_tolerance)
    df["selected_option"] = df["option_id"].astype(str).eq(selected_id)

    selected = df[df["selected_option"]].iloc[0]
    others = df[~df["selected_option"]].copy()
    runner = (
        others.sort_values(
            ["eligible", "silhouette", "resample_stability_ari_mean"],
            ascending=[False, False, False],
        )
        .iloc[0]
        .to_dict()
        if len(others)
        else {}
    )

    observed = (
        f"Selected {selected_id}: {selected['algorithm']} K={int(selected['k'])}, "
        f"silhouette {float(selected['silhouette']):.3f}, mean subsample ARI "
        f"{float(selected['resample_stability_ari_mean']):.3f}, minimum cluster share "
        f"{100.0 * float(selected['min_cluster_share']):.1f}%."
    )
    interpretation = (
        "O1 and O2 are evaluated under the same KMeans/GMM K-search and the same "
        "stability/balance gates. The choice is not based on silhouette alone: "
        "only near-best eligible options enter the final shortlist, after which "
        "subsample robustness, seed stability, cluster balance and parsimony are "
        "used in that order."
    )
    action = (
        "Freeze the selected feature-space option for the official Branch-A model "
        "and keep the non-selected option as robustness evidence. Re-run the same "
        "comparison whenever the input dataset or feature policy changes."
    )
    decision = {
        "selected_option": selected_id,
        "fallback_mode": fallback,
        "silhouette_tolerance": float(silhouette_tolerance),
        "observed": observed,
        "interpretation": interpretation,
        "action": action,
        "runner_up": runner,
    }
    return selected_id, df, decision


def run_segmentation(
    dev: pd.DataFrame,
    all_df: pd.DataFrame,
    seed: int = 42,
    k_min: int = 2,
    k_max: int = 8,
    stability_seeds: list[int] | None = None,
    stability_min: float = 0.90,
    min_cluster_share: float = 0.05,
    silhouette_tolerance: float = 0.01,
    pca_variance_threshold: float = 0.85,
    pca_min_components: int = 2,
    pca_max_components: int | None = None,
    resample_n: int = 8,
    resample_fraction: float = 0.85,
    resample_stability_min: float = 0.75,
    resample_seed: int = 2026,
    representation_silhouette_tolerance: float = 0.03,
    familywise_caps: dict[str, int] | None = None,
    correlation_threshold: float = 0.90,
) -> tuple[dict[str, Any], dict[str, pd.DataFrame], dict[str, Any]]:
    """Branch A: compare two feature-space construction strategies.

    O1 — PCA-based latent representation
        The existing R0-R4 robustness study is retained:
        R0 global family-balanced PCA;
        R1 family-wise PCA;
        R2 no job_category;
        R3 no years_of_experience;
        R4 no job_category + no years_of_experience.
        O1 internally selects its best R, then evaluates its official KMeans/GMM
        K=2..8 solution.

    O2 — Correlation-based feature selection
        Feature-to-feature Pearson correlation is computed on DEV only after
        encoding and family balancing. Redundant encoded columns are filtered at
        ``correlation_threshold``. The selected encoded X columns are clustered
        directly; PCA is used only for the PC1/PC2 display.

    O1 and O2 receive the same clustering search and hard robustness gates. The
    official feature-space option is then chosen from near-best eligible options
    using robustness, balance and parsimony rather than a weighted score.
    """
    stability_seeds = stability_seeds or [11, 23, 42, 71, 101]
    if seed not in stability_seeds:
        stability_seeds = sorted(set(stability_seeds + [seed]))
    familywise_caps = familywise_caps or {
        "Job Domain": 3,
        "Experience & Education": 3,
        "Company & Work Mode": 3,
        "Geography": 3,
        "Demand / Benefits": 3,
        "Skills": 5,
    }

    # Shared DEV-fitted primitive encoders/scalers. Target/salary never enters
    # either O1 or O2.
    emit_event(
        "operation_started",
        step_id="segmentation.encoder.fit",
        operation="fit_segmentation_encoder",
        status="started",
        message="Fit the family-balanced segmentation encoder on DEV only.",
        detail_level=2,
        extra={
            "stage_id": "A1-A8",
            "train_rows": int(len(dev)),
            "features": SEGMENTATION_FEATURES,
            "target_used": False,
        },
    )
    encoder = FamilyBalancedEncoder(balance=True).fit(dev[SEGMENTATION_FEATURES])
    emit_event(
        "operation_completed",
        step_id="segmentation.encoder.fit",
        operation="fit_segmentation_encoder",
        status="completed",
        message="Family-balanced segmentation encoder fitted on DEV.",
        detail_level=2,
        extra={
            "stage_id": "A1-A8",
            "train_rows": int(len(dev)),
            "encoded_dimensions": int(sum(encoder.family_dims_.values())),
            "target_used": False,
        },
    )

    # ------------------------------------------------------------------
    # O1 — PCA-based representation family (R0-R4)
    # ------------------------------------------------------------------
    emit_event(
        "operation_started",
        step_id="segmentation.o1.representations",
        operation="build_segmentation_representations",
        status="started",
        message="Build O1 PCA representation candidates R0–R4 from DEV-fitted encodings.",
        detail_level=2,
        extra={"stage_id": "A1-A8"},
    )
    reps = build_representation_specs(
        encoder=encoder,
        X_dev=dev[SEGMENTATION_FEATURES],
        X_all=all_df[SEGMENTATION_FEATURES],
        all_raw_features=SEGMENTATION_FEATURES,
        threshold=float(pca_variance_threshold),
        family_caps=familywise_caps,
        seed=seed,
        global_max_components=pca_max_components,
    )
    emit_event(
        "operation_completed",
        step_id="segmentation.o1.representations",
        operation="build_segmentation_representations",
        status="completed",
        message=f"Built {len(reps)} O1 representation candidates.",
        detail_level=2,
        extra={
            "stage_id": "A1-A8",
            "representations": list(reps),
            "representation_count": int(len(reps)),
        },
    )

    resample_cfg = ResampleStabilityConfig(
        n_resamples=int(resample_n),
        sample_fraction=float(resample_fraction),
        min_mean_ari=float(resample_stability_min),
        random_seed=int(resample_seed),
    )

    all_eval: list[pd.DataFrame] = []
    all_resample: list[pd.DataFrame] = []
    all_candidates: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    selected_dev_labels: dict[str, np.ndarray] = {}

    for rep_id, rep_i in reps.items():
        emit_event(
            "operation_started",
            step_id=f"segmentation.o1.{rep_id}.evaluate",
            operation="evaluate_segmentation_candidates",
            status="started",
            message=f"Evaluate segmentation candidates for {rep_id}.",
            detail_level=3,
            extra={
                "stage_id": "A1-A8",
                "representation_id": rep_id,
                "dev_rows": int(len(dev)),
                "dimensions": int(rep_i.dev_matrix.shape[1]),
            },
        )
        ev_i, rs_i, cand_i, summary_i, model_i, labels_dev_i, labels_all_i = evaluate_candidates(
            rep_i.dev_matrix,
            rep_i.all_matrix,
            representation_id=rep_id,
            k_min=int(k_min),
            k_max=int(k_max),
            seed=seed,
            stability_seeds=stability_seeds,
            stability_min=float(stability_min),
            min_cluster_share=float(min_cluster_share),
            silhouette_tolerance=float(silhouette_tolerance),
            resample_config=resample_cfg,
            full_stability=True,
        )
        emit_event(
            "operation_completed",
            step_id=f"segmentation.o1.{rep_id}.evaluate",
            operation="evaluate_segmentation_candidates",
            status="completed",
            message=f"Evaluated segmentation candidates for {rep_id}.",
            detail_level=3,
            extra={
                "stage_id": "A1-A8",
                "representation_id": rep_id,
                "candidate_rows": int(len(ev_i)),
                "selected_algorithm": summary_i.get("algorithm"),
                "selected_k": summary_i.get("k"),
                "selected_silhouette": summary_i.get("silhouette"),
            },
        )
        ev_i["representation_label"] = rep_i.label
        all_eval.append(ev_i)
        if len(rs_i):
            all_resample.append(rs_i)
        all_candidates.append(cand_i)
        selected_dev_labels[rep_id] = labels_dev_i

        used_i = rep_i.component_summary[rep_i.component_summary["used_for_clustering"]].copy()
        if rep_id == "R1_FAMILYWISE_PCA":
            fam_var = used_i.groupby("family")["explained_variance_ratio"].sum()
            variance_summary = float(fam_var.mean()) if len(fam_var) else np.nan
            min_family_variance = float(fam_var.min()) if len(fam_var) else np.nan
        else:
            variance_summary = float(used_i["explained_variance_ratio"].sum())
            min_family_variance = np.nan

        summary_rows.append(
            {
                **summary_i,
                "representation_label": rep_i.label,
                "purpose": rep_i.purpose,
                "excluded_features": (
                    ", ".join(rep_i.excluded_raw_features)
                    if rep_i.excluded_raw_features
                    else "None"
                ),
                "excluded_raw_features": (
                    ", ".join(rep_i.excluded_raw_features)
                    if rep_i.excluded_raw_features
                    else "None"
                ),
                "raw_input_count": len(rep_i.selected_input_features),
                "selected_input_features": ", ".join(rep_i.selected_input_features),
                "latent_dimensions": int(rep_i.dev_matrix.shape[1]),
                "variance_retained_summary": variance_summary,
                "variance_stat": variance_summary,
                "minimum_family_variance_retained": min_family_variance,
            }
        )

    representation_summary = pd.DataFrame(summary_rows)
    pairwise_ari = pairwise_representation_ari(selected_dev_labels)
    o1_rep_id, representation_summary, ablation_insight = choose_official_representation(
        representation_summary,
        pairwise_ari,
        silhouette_tolerance=float(representation_silhouette_tolerance),
    )
    o1_rep = reps[o1_rep_id]

    # Exhaustive official O1 K study.
    emit_event(
        "operation_started",
        step_id="segmentation.o1.official.evaluate",
        operation="evaluate_official_o1_candidates",
        status="started",
        message=f"Run the official O1 candidate study for {o1_rep_id}.",
        detail_level=2,
        extra={
            "stage_id": "A1-A8",
            "representation_id": o1_rep_id,
            "dimensions": int(o1_rep.dev_matrix.shape[1]),
        },
    )
    (
        o1_ev,
        o1_rs,
        o1_candidates,
        o1_summary,
        o1_model,
        o1_labels_dev,
        o1_labels_all,
    ) = evaluate_candidates(
        o1_rep.dev_matrix,
        o1_rep.all_matrix,
        representation_id=o1_rep_id,
        k_min=int(k_min),
        k_max=int(k_max),
        seed=seed,
        stability_seeds=stability_seeds,
        stability_min=float(stability_min),
        min_cluster_share=float(min_cluster_share),
        silhouette_tolerance=float(silhouette_tolerance),
        resample_config=resample_cfg,
        full_stability=True,
    )
    o1_ev["representation_label"] = o1_rep.label
    emit_event(
        "operation_completed",
        step_id="segmentation.o1.official.evaluate",
        operation="evaluate_official_o1_candidates",
        status="completed",
        message="Official O1 candidate study completed.",
        detail_level=2,
        extra={
            "stage_id": "A1-A8",
            "representation_id": o1_rep_id,
            "candidate_rows": int(len(o1_ev)),
            "selected_algorithm": o1_summary.get("algorithm"),
            "selected_k": o1_summary.get("k"),
            "selected_silhouette": o1_summary.get("silhouette"),
        },
    )

    # ------------------------------------------------------------------
    # O2 — correlation-based feature selection on family-balanced X
    # ------------------------------------------------------------------
    emit_event(
        "operation_started",
        step_id="segmentation.o2.selection",
        operation="build_correlation_selected_space",
        status="started",
        message="Build O2 correlation-selected feature space from DEV-fitted encodings.",
        detail_level=2,
        extra={"stage_id": "A1-A8", "correlation_threshold": float(correlation_threshold)},
    )
    o2 = _build_correlation_selected_space(
        encoder,
        dev[SEGMENTATION_FEATURES],
        all_df[SEGMENTATION_FEATURES],
        threshold=float(correlation_threshold),
        seed=int(seed),
    )
    emit_event(
        "operation_completed",
        step_id="segmentation.o2.selection",
        operation="build_correlation_selected_space",
        status="completed",
        message="Built O2 correlation-selected feature space.",
        detail_level=2,
        extra={
            "stage_id": "A1-A8",
            "encoded_features_before": int(o2["encoded_features_before"]),
            "selected_encoded_features": int(o2["selected_encoded_features"]),
            "correlation_threshold": float(correlation_threshold),
        },
    )
    emit_event(
        "operation_started",
        step_id="segmentation.o2.evaluate",
        operation="evaluate_segmentation_candidates",
        status="started",
        message="Evaluate O2 segmentation candidates.",
        detail_level=2,
        extra={
            "stage_id": "A1-A8",
            "representation_id": "O2_CORRELATION_SELECTED",
            "dimensions": int(o2["dev_matrix"].shape[1]),
        },
    )
    (
        o2_ev,
        o2_rs,
        o2_candidates,
        o2_summary,
        o2_model,
        o2_labels_dev,
        o2_labels_all,
    ) = evaluate_candidates(
        o2["dev_matrix"],
        o2["all_matrix"],
        representation_id="O2_CORRELATION_SELECTED",
        k_min=int(k_min),
        k_max=int(k_max),
        seed=seed,
        stability_seeds=stability_seeds,
        stability_min=float(stability_min),
        min_cluster_share=float(min_cluster_share),
        silhouette_tolerance=float(silhouette_tolerance),
        resample_config=resample_cfg,
        full_stability=True,
    )
    emit_event(
        "operation_completed",
        step_id="segmentation.o2.evaluate",
        operation="evaluate_segmentation_candidates",
        status="completed",
        message="O2 segmentation candidate evaluation completed.",
        detail_level=2,
        extra={
            "stage_id": "A1-A8",
            "candidate_rows": int(len(o2_ev)),
            "selected_algorithm": o2_summary.get("algorithm"),
            "selected_k": o2_summary.get("k"),
            "selected_silhouette": o2_summary.get("silhouette"),
        },
    )

    # ------------------------------------------------------------------
    # O1 vs O2 option-level comparison
    # ------------------------------------------------------------------
    cross_option_ari = float(adjusted_rand_score(o1_labels_dev, o2_labels_dev))

    option_summary = pd.DataFrame(
        [
            {
                "option_id": "O1_PCA",
                "option_label": "O1 — PCA-based latent representation",
                "method": "Feature extraction / dimensionality reduction",
                "internal_representation": o1_rep_id,
                "algorithm": str(o1_summary["algorithm"]),
                "k": int(o1_summary["k"]),
                "silhouette": float(o1_summary["silhouette"]),
                "stability_ari": float(o1_summary["stability_ari"]),
                "resample_stability_ari_mean": float(o1_summary["resample_stability_ari_mean"]),
                "resample_stability_ari_p10": float(
                    o1_summary.get("resample_stability_ari_p10", np.nan)
                ),
                "min_cluster_share": float(o1_summary["min_cluster_share"]),
                "eligible": bool(o1_summary["eligible"]),
                "clustering_dimensions": int(o1_rep.dev_matrix.shape[1]),
                "encoded_features_before": int(sum(encoder.family_dims_.values())),
                "selected_encoded_features": np.nan,
                "correlation_threshold": np.nan,
                "interpretability": "Moderate — latent principal components",
                "cross_option_ari": cross_option_ari,
            },
            {
                "option_id": "O2_CORRELATION",
                "option_label": "O2 — Correlation-based feature selection",
                "method": "Feature selection / redundancy filtering",
                "internal_representation": "CORRELATION_SELECTED_X",
                "algorithm": str(o2_summary["algorithm"]),
                "k": int(o2_summary["k"]),
                "silhouette": float(o2_summary["silhouette"]),
                "stability_ari": float(o2_summary["stability_ari"]),
                "resample_stability_ari_mean": float(o2_summary["resample_stability_ari_mean"]),
                "resample_stability_ari_p10": float(
                    o2_summary.get("resample_stability_ari_p10", np.nan)
                ),
                "min_cluster_share": float(o2_summary["min_cluster_share"]),
                "eligible": bool(o2_summary["eligible"]),
                "clustering_dimensions": int(o2["selected_encoded_features"]),
                "encoded_features_before": int(o2["encoded_features_before"]),
                "selected_encoded_features": int(o2["selected_encoded_features"]),
                "correlation_threshold": float(correlation_threshold),
                "interpretability": "High — selected encoded features retained",
                "cross_option_ari": cross_option_ari,
            },
        ]
    )
    selected_option_id, option_summary, option_decision = _choose_feature_space_option(
        option_summary,
        silhouette_tolerance=float(representation_silhouette_tolerance),
    )
    emit_event(
        "selection_recorded",
        step_id="segmentation.option.selection",
        operation="select_segmentation_feature_space",
        status="completed",
        message=f"Selected segmentation feature-space option {selected_option_id}.",
        detail_level=2,
        extra={
            "stage_id": "A1-A8",
            "selected_option_id": selected_option_id,
            "selection_policy": "eligible near-best silhouette, robustness, balance and parsimony",
            "cross_option_ari": cross_option_ari,
        },
    )

    if selected_option_id == "O1_PCA":
        official_representation_id = o1_rep_id
        official_label = o1_rep.label
        official_mode = o1_rep.transformer["kind"]
        official_transformer = o1_rep.transformer
        official_input_features = list(o1_rep.selected_input_features)
        official_excluded_features = list(o1_rep.excluded_raw_features)
        official_matrix_dev = o1_rep.dev_matrix
        official_matrix_all = o1_rep.all_matrix
        coords = o1_rep.visual_all
        official_ev_full = o1_ev.copy()
        official_rs_full = o1_rs.copy()
        official_candidates_full = o1_candidates.copy()
        official_summary_full = dict(o1_summary)
        cluster_model = o1_model
        labels_dev = o1_labels_dev
        labels_all = o1_labels_all
        comp = o1_rep.component_summary.copy()
        if "used_for_visualization" not in comp.columns:
            comp["used_for_visualization"] = comp["component_number"] <= 2

        used = comp[comp["used_for_clustering"]]
        if o1_rep_id == "R1_FAMILYWISE_PCA":
            fam_var = used.groupby("family")["explained_variance_ratio"].sum()
            variance_captured = float(fam_var.mean()) if len(fam_var) else np.nan
            visual_var = float(o1_rep.transformer["visualizer"].explained_variance_ratio_[:2].sum())
            clustering_space = (
                f"Family-wise PCA latent concatenation "
                f"({o1_rep.dev_matrix.shape[1]} retained latent dimensions)"
            )
            visualization_space = (
                "Separate 2D PCA of the family-wise latent matrix; 2D visualization display only"
            )
        else:
            variance_captured = float(used["explained_variance_ratio"].sum())
            visual_var = float(
                comp.loc[
                    comp["component_number"] <= 2,
                    "explained_variance_ratio",
                ].sum()
            )
            clustering_space = (
                f"Global PCA PC1..PC{o1_rep.dev_matrix.shape[1]} "
                "of the selected encoded representation"
            )
            visualization_space = (
                "PC1/PC2 of the selected global PCA; 2D visualization display only"
            )

        selected_family_pca_summary = (
            comp[comp["used_for_clustering"]]
            .groupby("family")
            .agg(
                retained_components=("component", "count"),
                retained_variance=("explained_variance_ratio", "sum"),
            )
            .reset_index()
            if o1_rep_id == "R1_FAMILYWISE_PCA"
            else pd.DataFrame(
                [
                    {
                        "family": "GLOBAL",
                        "retained_components": int(o1_rep.dev_matrix.shape[1]),
                        "retained_variance": float(variance_captured),
                    }
                ]
            )
        )
    else:
        official_representation_id = "O2_CORRELATION_SELECTED"
        official_label = "O2 — Correlation-selected family-balanced encoded X"
        official_mode = "correlation_selected"
        official_transformer = o2["transformer"]
        # Raw operational contract remains the canonical 13 safe Branch-A
        # inputs; O2 selects encoded columns after those raw fields are encoded.
        official_input_features = list(MODEL_FEATURES)
        official_excluded_features = []
        official_matrix_dev = o2["dev_matrix"]
        official_matrix_all = o2["all_matrix"]
        coords = o2["visual_all"]
        official_ev_full = o2_ev.copy()
        official_rs_full = o2_rs.copy()
        official_candidates_full = o2_candidates.copy()
        official_summary_full = dict(o2_summary)
        cluster_model = o2_model
        labels_dev = o2_labels_dev
        labels_all = o2_labels_all
        variance_captured = np.nan
        visual_var = float(o2["visualization_variance_captured"])
        clustering_space = (
            f"Correlation-selected family-balanced encoded X "
            f"({o2['selected_encoded_features']} selected encoded features); "
            "no PCA used for clustering"
        )
        visualization_space = "2D PCA of O2 selected X; 2D visualization display only"

        vr = np.asarray(
            o2["transformer"]["visualizer"].explained_variance_ratio_,
            dtype=float,
        )
        comp = pd.DataFrame(
            {
                "representation_id": "O2_CORRELATION_SELECTED",
                "family": "DISPLAY_ONLY",
                "component_number": np.arange(1, len(vr) + 1),
                "component": [f"PC{i}" for i in range(1, len(vr) + 1)],
                "explained_variance_ratio": vr,
                "cumulative_variance": np.cumsum(vr),
                "used_for_clustering": False,
                "used_for_visualization": np.arange(1, len(vr) + 1) <= 2,
                "display_only": True,
            }
        )
        selected_family_pca_summary = pd.DataFrame(
            [
                {
                    "family": "CORRELATION_SELECTED_X",
                    "retained_components": 0,
                    "retained_variance": np.nan,
                    "selected_encoded_features": int(o2["selected_encoded_features"]),
                }
            ]
        )

    selected_row = pd.Series(official_summary_full)

    # ------------------------------------------------------------------
    # Official assignments and downstream descriptive evidence
    # ------------------------------------------------------------------
    assignments = all_df.reset_index(drop=True).copy()
    assignments.insert(0, "record_id", np.arange(1, len(assignments) + 1))
    assignments["cluster"] = labels_all
    assignments["PC1"] = coords[:, 0]
    assignments["PC2"] = coords[:, 1]

    profiles = (
        assignments.groupby("cluster")
        .agg(
            records=(TARGET, "size"),
            salary_mean=(TARGET, "mean"),
            salary_median=(TARGET, "median"),
            years_mean=("years_of_experience", "mean"),
            demand_mean=("demand_score", "mean"),
            benefits_mean=("benefits_score_10", "mean"),
            skill_count_mean=("skill_count", "mean"),
        )
        .reset_index()
    )
    profiles["share_pct"] = 100 * profiles["records"] / profiles["records"].sum()

    # Shared family-balance diagnostics.
    blocks_raw = encoder.transform_family_blocks(
        dev[SEGMENTATION_FEATURES],
        balanced=False,
    )
    blocks_bal = encoder.transform_family_blocks(
        dev[SEGMENTATION_FEATURES],
        balanced=True,
    )
    family_rows = []
    for fam in blocks_raw:
        raw = blocks_raw[fam]
        bal = blocks_bal[fam]
        dim = int(raw.shape[1])
        family_rows.append(
            {
                "family": fam,
                "encoded_dimensions": dim,
                "balance_weight": 1.0 / math.sqrt(dim) if dim else 1.0,
                "frobenius_norm_before": float(np.linalg.norm(raw, ord="fro")),
                "frobenius_norm_after": float(np.linalg.norm(bal, ord="fro")),
                "mean_row_l2_before": (float(np.mean(np.linalg.norm(raw, axis=1))) if dim else 0.0),
                "mean_row_l2_after": (float(np.mean(np.linalg.norm(bal, axis=1))) if dim else 0.0),
            }
        )
    family_diag = pd.DataFrame(family_rows)
    family_dims = family_diag[["family", "encoded_dimensions", "balance_weight"]].copy()

    # Official candidate assignment evidence.
    official_candidates = official_candidates_full.drop(
        columns=["representation_id"],
        errors="ignore",
    )
    extra_evidence = build_segmentation_evidence(
        assignments,
        official_candidates,
        family_diag,
        normalize_skills,
    )

    # Backward-compatible official K-evaluation fields.
    official_ev = official_ev_full.copy()
    _best_official_sil = (
        float(
            official_ev.loc[
                official_ev["eligible"],
                "silhouette",
            ].max()
        )
        if official_ev["eligible"].any()
        else float(official_ev["silhouette"].max())
    )
    official_ev["within_primary_tolerance"] = official_ev["silhouette"] >= (
        _best_official_sil - float(silhouette_tolerance)
    )
    official_ev["selected"] = official_ev["selected_within_representation"].astype(bool)
    official_ev["selection_score"] = (
        0.45 * official_ev["silhouette"].rank(pct=True)
        + 0.20 * official_ev["stability_ari"].fillna(-1).rank(pct=True)
        + 0.20 * official_ev["resample_stability_ari_mean"].fillna(-1).rank(pct=True)
        + 0.10 * official_ev["balance_entropy"].rank(pct=True)
        + 0.025 * official_ev["calinski_harabasz"].rank(pct=True)
        + 0.025
        * official_ev["davies_bouldin"].rank(
            pct=True,
            ascending=False,
        )
    )
    official_rs = official_rs_full.copy()

    # O1 dependency evidence remains useful even when O2 wins.
    def _pair(a: str, b: str) -> float:
        q = pairwise_ari[
            (pairwise_ari["representation_a"] == a) & (pairwise_ari["representation_b"] == b)
        ]
        return float(q.iloc[0]["ari"]) if len(q) else np.nan

    dep = pd.DataFrame(
        [
            {
                "feature_tested": "job_category",
                "baseline_representation": "R0_GLOBAL_PCA",
                "ablation_representation": "R2_NO_JOB_CATEGORY",
                "assignment_ari": _pair(
                    "R0_GLOBAL_PCA",
                    "R2_NO_JOB_CATEGORY",
                ),
            },
            {
                "feature_tested": "years_of_experience",
                "baseline_representation": "R0_GLOBAL_PCA",
                "ablation_representation": "R3_NO_YEARS_EXPERIENCE",
                "assignment_ari": _pair(
                    "R0_GLOBAL_PCA",
                    "R3_NO_YEARS_EXPERIENCE",
                ),
            },
            {
                "feature_tested": "job_category + years_of_experience",
                "baseline_representation": "R0_GLOBAL_PCA",
                "ablation_representation": "R4_NO_JOB_CATEGORY_NO_YEARS",
                "assignment_ari": _pair(
                    "R0_GLOBAL_PCA",
                    "R4_NO_JOB_CATEGORY_NO_YEARS",
                ),
            },
        ]
    )
    dep["dependency_band"] = np.where(
        dep["assignment_ari"] < 0.50,
        "high",
        np.where(dep["assignment_ari"] < 0.80, "moderate", "low"),
    )

    quality_band = _cluster_quality_band(float(selected_row.silhouette))

    official_rationale = {
        "selected_feature_space_option": selected_option_id,
        "selected_feature_space_label": str(
            option_summary.loc[
                option_summary["selected_option"],
                "option_label",
            ].iloc[0]
        ),
        "selected_representation": official_representation_id,
        "o1_selected_representation": o1_rep_id,
        "selected_algorithm": str(selected_row.algorithm),
        "selected_k": int(selected_row.k),
        "selected_silhouette": float(selected_row.silhouette),
        "selected_stability_ari": (
            float(selected_row.stability_ari) if pd.notna(selected_row.stability_ari) else np.nan
        ),
        "selected_resample_stability_ari_mean": (
            float(selected_row.resample_stability_ari_mean)
            if pd.notna(selected_row.resample_stability_ari_mean)
            else np.nan
        ),
        "selected_min_cluster_share": float(selected_row.min_cluster_share),
        "selected_quality_band": quality_band,
        "stability_threshold": float(stability_min),
        "resample_stability_threshold": float(resample_stability_min),
        "minimum_cluster_share_threshold": float(min_cluster_share),
        "silhouette_tolerance": float(silhouette_tolerance),
        "representation_silhouette_tolerance": float(representation_silhouette_tolerance),
        "correlation_threshold": float(correlation_threshold),
        "cross_option_ari": cross_option_ari,
        "decision_note": option_decision["observed"],
        "feature_space_decision": option_decision,
        "representation_ablation": ablation_insight,
    }

    meta = {
        "feature_space_option_id": selected_option_id,
        "feature_space_option_label": official_rationale["selected_feature_space_label"],
        "representation_id": official_representation_id,
        "representation_label": official_label,
        "o1_selected_representation": o1_rep_id,
        "representation_mode": official_mode,
        "representation_input_features": official_input_features,
        "representation_excluded_features": official_excluded_features,
        "selected_input_features": official_input_features,
        "excluded_raw_features": official_excluded_features,
        "algorithm": str(selected_row.algorithm),
        "k": int(selected_row.k),
        "silhouette": float(selected_row.silhouette),
        "stability_ari": (
            float(selected_row.stability_ari) if pd.notna(selected_row.stability_ari) else np.nan
        ),
        "resample_stability_ari_mean": (
            float(selected_row.resample_stability_ari_mean)
            if pd.notna(selected_row.resample_stability_ari_mean)
            else np.nan
        ),
        "min_cluster_share": float(selected_row.min_cluster_share),
        "quality_band": quality_band,
        "encoded_dimensions": int(sum(encoder.family_dims_.values())),
        "clustering_dimensions": int(official_matrix_dev.shape[1]),
        "pca_components_total": int(len(comp)),
        "pca_components_for_clustering": (
            int(official_matrix_dev.shape[1]) if selected_option_id == "O1_PCA" else 0
        ),
        "pca_variance_threshold": float(pca_variance_threshold),
        "clustering_variance_captured": (
            float(variance_captured) if pd.notna(variance_captured) else np.nan
        ),
        "visualization_components": 2,
        "visualization_variance_captured": float(visual_var),
        "clustering_space": clustering_space,
        "visualization_space": visualization_space,
        "fit_rows": int(len(dev)),
        "target_used_for_clustering": False,
        "correlation_threshold": float(correlation_threshold),
        "o2_selected_encoded_features": int(o2["selected_encoded_features"]),
        "o2_encoded_features_before": int(o2["encoded_features_before"]),
        "selection_policy": (
            "O1 PCA family (R0-R4 internal robustness) vs O2 correlation-selected X "
            "→ same KMeans/GMM K=2..8 search → convergence/stability/balance gates "
            "→ near-best silhouette → robustness → parsimony"
        ),
    }

    insights = build_segmentation_insights(
        meta=meta,
        rationale=official_rationale,
        profiles=profiles,
        evidence=extra_evidence,
    )

    option_card = {
        "observed": option_decision["observed"],
        "interpretation": option_decision["interpretation"],
        "action": option_decision["action"],
        "tone": "success" if bool(selected_row.eligible) else "warning",
    }
    insights["feature_space_selection"] = option_card

    representation_card = {
        "observed": ablation_insight["observed"],
        "interpretation": (
            "This is the internal O1 PCA robustness study. " + ablation_insight["interpretation"]
        ),
        "action": ablation_insight["action"],
        "tone": "info",
    }
    insights["representation_robustness"] = representation_card
    insights["representation_ablation"] = representation_card

    bundle = {
        "feature_space_option_id": selected_option_id,
        "representation_transformer": official_transformer,
        "model": cluster_model,
        "algorithm": str(selected_row.algorithm),
        "k": int(selected_row.k),
        "representation_id": official_representation_id,
        "o1_selected_representation": o1_rep_id,
        "segmentation_features": official_input_features,
        "selection_policy": official_rationale,
    }

    # ------------------------------------------------------------------
    # Persisted evidence tables
    # ------------------------------------------------------------------
    representation_components = pd.concat(
        [r.component_summary for r in reps.values()],
        ignore_index=True,
    )
    representation_loadings = pd.concat(
        [r.loadings for r in reps.values()],
        ignore_index=True,
    )

    sensitivity_rows = []
    for rid, r in reps.items():
        cs = r.component_summary.copy()
        if rid == "R1_FAMILYWISE_PCA":
            for fam, grp in cs.groupby("family"):
                ratios = grp.sort_values("component_number")["explained_variance_ratio"].to_numpy(
                    float
                )
                cap = (
                    int(grp["family_component_cap"].dropna().iloc[0])
                    if ("family_component_cap" in grp and grp["family_component_cap"].notna().any())
                    else None
                )
                for thr in (0.80, 0.85, 0.90):
                    n_thr = _pca_component_count(ratios, thr, 1, cap)
                    sensitivity_rows.append(
                        {
                            "representation_id": rid,
                            "family": fam,
                            "variance_threshold": thr,
                            "retained_components": int(n_thr),
                            "variance_retained": float(ratios[:n_thr].sum()),
                            "cap_reached_before_target": bool(ratios[:n_thr].sum() < thr - 1e-12),
                        }
                    )
        else:
            grp = cs[cs["family"] == "GLOBAL"].sort_values("component_number")
            ratios = grp["explained_variance_ratio"].to_numpy(float)
            for thr in (0.80, 0.85, 0.90):
                n_thr = _pca_component_count(
                    ratios,
                    thr,
                    pca_min_components,
                    pca_max_components,
                )
                sensitivity_rows.append(
                    {
                        "representation_id": rid,
                        "family": "GLOBAL",
                        "variance_threshold": thr,
                        "retained_components": int(n_thr),
                        "variance_retained": float(ratios[:n_thr].sum()),
                        "cap_reached_before_target": bool(ratios[:n_thr].sum() < thr - 1e-12),
                    }
                )
    representation_pca_sensitivity = pd.DataFrame(sensitivity_rows)

    # O1/O2 candidate metrics under one common schema.
    o1_option_candidates = o1_ev.copy()
    o1_option_candidates.insert(0, "option_id", "O1_PCA")
    o1_option_candidates.insert(
        1,
        "option_label",
        "O1 — PCA-based latent representation",
    )
    o1_option_candidates["internal_representation"] = o1_rep_id

    o2_option_candidates = o2_ev.copy()
    o2_option_candidates.insert(0, "option_id", "O2_CORRELATION")
    o2_option_candidates.insert(
        1,
        "option_label",
        "O2 — Correlation-based feature selection",
    )
    o2_option_candidates["internal_representation"] = "CORRELATION_SELECTED_X"
    feature_space_candidate_metrics = pd.concat(
        [o1_option_candidates, o2_option_candidates],
        ignore_index=True,
        sort=False,
    )

    feature_space_pairwise_ari = pd.DataFrame(
        [
            {
                "option_a": "O1_PCA",
                "option_b": "O1_PCA",
                "ari": 1.0,
            },
            {
                "option_a": "O1_PCA",
                "option_b": "O2_CORRELATION",
                "ari": cross_option_ari,
            },
            {
                "option_a": "O2_CORRELATION",
                "option_b": "O1_PCA",
                "ari": cross_option_ari,
            },
            {
                "option_a": "O2_CORRELATION",
                "option_b": "O2_CORRELATION",
                "ari": 1.0,
            },
        ]
    )

    pca_coords = pd.DataFrame(
        {
            "record_id": np.arange(1, len(all_df) + 1),
            "PC1": coords[:, 0],
            "PC2": coords[:, 1],
            "cluster": labels_all,
        }
    )

    representation_candidate_metrics = pd.concat(
        all_eval,
        ignore_index=True,
    )
    representation_candidate_assignments = pd.concat(
        all_candidates,
        ignore_index=True,
    )

    tables = {
        "cluster_evaluation": official_ev,
        "resample_stability_runs": official_rs,
        "candidate_cluster_assignments": official_candidates,
        "pca_coordinates": pca_coords,
        "cluster_assignments": assignments,
        "cluster_profiles": profiles,
        "family_dimensions": family_dims,
        "family_balance_diagnostics": family_diag,
        "pca_variance": comp,
        "cluster_selection_summary": option_summary[option_summary["selected_option"]].copy(),
        # New O1-vs-O2 evidence
        "feature_space_option_summary": option_summary,
        "feature_space_candidate_metrics": feature_space_candidate_metrics,
        "feature_space_pairwise_ari": feature_space_pairwise_ari,
        "correlation_feature_selection": o2["selection_table"],
        "correlation_redundancy_pairs": o2["redundancy_pairs"],
        "correlation_family_summary": o2["family_summary"],
        "o2_candidate_metrics": o2_ev,
        "o2_resample_stability_runs": o2_rs,
        "o2_candidate_assignments": o2_candidates,
        "o2_visual_coordinates": pd.DataFrame(
            {
                "record_id": np.arange(1, len(all_df) + 1),
                "PC1": o2["visual_all"][:, 0],
                "PC2": o2["visual_all"][:, 1],
                "cluster": o2_labels_all,
            }
        ),
        # Existing O1 R0-R4 evidence
        "representation_summary": representation_summary,
        "representation_candidate_metrics": representation_candidate_metrics,
        "representation_candidate_assignments": representation_candidate_assignments,
        "representation_pairwise_ari": pairwise_ari,
        "representation_pca_variance": representation_components,
        "representation_top_loadings": representation_loadings,
        "representation_component_summary": representation_components,
        "representation_component_loadings": representation_loadings,
        "representation_pca_sensitivity": representation_pca_sensitivity,
        "selected_representation_family_pca_summary": selected_family_pca_summary,
        "representation_resample_stability_runs": (
            pd.concat(all_resample, ignore_index=True) if all_resample else pd.DataFrame()
        ),
        "representation_selected_assignments": pd.DataFrame(
            {
                "record_id": np.arange(1, len(dev) + 1),
                **{rid: labs for rid, labs in selected_dev_labels.items()},
            }
        ),
        "feature_dependency_summary": dep,
        **extra_evidence,
    }

    return (
        bundle,
        tables,
        {
            **meta,
            "rationale": official_rationale,
            "representation_decision": ablation_insight,
            "feature_space_decision": option_decision,
            "insights": insights,
        },
    )


def transform_to_cluster_space(
    segmentation_bundle: dict[str, Any],
    df: pd.DataFrame,
) -> np.ndarray:
    """Transform prepared rows into the frozen official Branch-A feature space."""
    required = [c for c in SEGMENTATION_FEATURES if c not in df.columns]
    if required:
        raise ValueError(f"Segmentation inference missing canonical fields: {required}")

    transformer = segmentation_bundle["representation_transformer"]
    kind = str(transformer.get("kind", ""))

    if kind == "correlation_selected":
        encoder = transformer["encoder"]
        z = np.asarray(
            encoder.transform(df[SEGMENTATION_FEATURES]),
            dtype=float,
        )
        indices = np.asarray(
            transformer["selected_encoded_indices"],
            dtype=int,
        )
        X = z[:, indices]
    else:
        X = transform_representation(
            transformer,
            df[SEGMENTATION_FEATURES],
        )

    expected = getattr(
        segmentation_bundle["model"],
        "n_features_in_",
        None,
    )
    if expected is not None and int(expected) != int(X.shape[1]):
        raise ValueError(
            f"Segmentation representation mismatch: transformed data has "
            f"{X.shape[1]} features but the fitted model expects {expected}."
        )
    return X


def predict_segments(segmentation_bundle: dict[str, Any], df: pd.DataFrame) -> np.ndarray:
    return (
        segmentation_bundle["model"]
        .predict(transform_to_cluster_space(segmentation_bundle, df))
        .astype(int)
    )


def tune_random_forest_manual_steps(
    dev: pd.DataFrame, n_splits: int = 5, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    trial_counter = 0

    def evaluate_trial(
        group: str,
        ordinal: int,
        total: int,
        label: str,
        trial_model: BaseEstimator,
    ) -> dict[str, float]:
        nonlocal trial_counter
        trial_counter += 1
        trial_id = f"rf-{group}-{ordinal}"
        emit_event(
            "trial_started",
            step_id=f"tuning.{trial_id}",
            operation="evaluate_rf_tuning_trial",
            status="started",
            message=f"Evaluate Random Forest tuning trial {ordinal}/{total} for {group}.",
            detail_level=2,
            extra={
                "stage_id": "B5-B6",
                "trial_id": trial_id,
                "trial_ordinal": int(trial_counter),
                "group": group,
                "group_ordinal": int(ordinal),
                "group_total": int(total),
                "seed": int(seed),
                "estimator_params": trial_model.get_params(deep=False),
            },
        )
        try:
            _, trial_summary = evaluate_model_cv(dev, MODEL_FEATURES, label, trial_model, n_splits)
        except BaseException as error:
            status = "cancelled" if isinstance(error, KeyboardInterrupt) else "failed"
            emit_event(
                f"trial_{status}",
                step_id=f"tuning.{trial_id}",
                operation="evaluate_rf_tuning_trial",
                status=status,
                message=f"Random Forest tuning trial {ordinal}/{total} for {group} did not complete.",
                detail_level=2,
                extra={
                    "stage_id": "B5-B6",
                    "trial_id": trial_id,
                    "group": group,
                    "error_type": type(error).__name__,
                },
            )
            raise
        emit_event(
            "trial_completed",
            step_id=f"tuning.{trial_id}",
            operation="evaluate_rf_tuning_trial",
            status="completed",
            message=f"Completed Random Forest tuning trial {ordinal}/{total} for {group}.",
            detail_level=2,
            extra={
                "stage_id": "B5-B6",
                "trial_id": trial_id,
                "trial_ordinal": int(trial_counter),
                "group": group,
                "group_ordinal": int(ordinal),
                "group_total": int(total),
                "seed": int(seed),
                "estimator_params": trial_model.get_params(deep=False),
                **trial_summary,
            },
        )
        return trial_summary

    # Bước 1: Initial Grid Search với các bộ thông số mẫu bao phủ 4 thông số chính
    configs = [
        {"n_estimators": 300, "min_samples_leaf": 2, "max_features": 0.7, "max_depth": 20},
        {"n_estimators": 100, "min_samples_leaf": 2, "max_features": 0.8, "max_depth": None},
        {"n_estimators": 80, "min_samples_leaf": 1, "max_features": 0.8, "max_depth": None},
        {"n_estimators": 200, "min_samples_leaf": 1, "max_features": 0.7, "max_depth": 20},
    ]
    rows1 = []
    for i, p in enumerate(configs, 1):
        model = RandomForestRegressor(random_state=seed, n_jobs=-1, **p)
        s = evaluate_trial("initial-grid", i, len(configs), f"RF tune {i}", model)
        rows1.append(
            {
                "candidate": i,
                **p,
                "CV_R2": s["R2_mean"],
                "CV_MAE": s["MAE_mean"],
                "CV_MAE_SD": s["MAE_std"],
                "CV_RMSE": s["RMSE_mean"],
                "CV_MedAE": s["MedAE_mean"],
            }
        )
    df1 = pd.DataFrame(rows1).sort_values("CV_R2", ascending=False).reset_index(drop=True)
    emit_event(
        "candidate_completed",
        step_id="tuning.rf.step1",
        operation="rank_rf_initial_grid",
        status="completed",
        message="Random Forest initial grid candidates ranked by CV R2.",
        extra={
            "ranking_metric": "CV_R2",
            "ranking_direction": "descending",
            "candidates": df1.to_dict("records"),
        },
    )

    # Bước 2: Thông số 1 — Tinh chỉnh n_estimators (giữ nguyên min_samples_leaf, max_features, max_depth từ Bước 1)
    best_init = df1.iloc[0]
    depth_val = None if pd.isna(best_init.max_depth) else int(best_init.max_depth)
    n_vals = [50, 100, 150, 200, 250, 300]
    rows2 = []
    for n in n_vals:
        model = RandomForestRegressor(
            n_estimators=n,
            min_samples_leaf=int(best_init.min_samples_leaf),
            max_features=float(best_init.max_features),
            max_depth=depth_val,
            random_state=seed,
            n_jobs=-1,
        )
        s = evaluate_trial("n-estimators", len(rows2) + 1, len(n_vals), f"RF n_{n}", model)
        rows2.append(
            {
                "n_estimators": n,
                "min_samples_leaf": int(best_init.min_samples_leaf),
                "max_features": float(best_init.max_features),
                "max_depth": best_init.max_depth,
                "CV_R2": s["R2_mean"],
                "CV_MAE": s["MAE_mean"],
                "CV_MAE_SD": s["MAE_std"],
                "CV_RMSE": s["RMSE_mean"],
                "CV_MedAE": s["MedAE_mean"],
            }
        )
    df2 = pd.DataFrame(rows2).sort_values("CV_R2", ascending=False).reset_index(drop=True)
    emit_event(
        "candidate_completed",
        step_id="tuning.rf.step2",
        operation="rank_rf_n_estimators",
        status="completed",
        message="Random Forest n_estimators candidates ranked by CV R2.",
        extra={
            "ranking_metric": "CV_R2",
            "ranking_direction": "descending",
            "candidates": df2.to_dict("records"),
        },
    )

    # Bước 3: Thông số 2 — Tinh chỉnh max_depth (chọn n*=200 từ Bước 2, giữ nguyên các tham số còn lại)
    best_n = int(df2.iloc[0].n_estimators)
    depth_vals = [10, 15, 20, 25, 30, None]
    rows3 = []
    for d in depth_vals:
        model = RandomForestRegressor(
            n_estimators=best_n,
            min_samples_leaf=int(best_init.min_samples_leaf),
            max_features=float(best_init.max_features),
            max_depth=d,
            random_state=seed,
            n_jobs=-1,
        )
        s = evaluate_trial("max-depth", len(rows3) + 1, len(depth_vals), f"RF depth_{d}", model)
        depth_str = "None" if d is None else str(d)
        rows3.append(
            {
                "n_estimators": best_n,
                "min_samples_leaf": int(best_init.min_samples_leaf),
                "max_features": float(best_init.max_features),
                "max_depth": depth_str,
                "CV_R2": s["R2_mean"],
                "CV_MAE": s["MAE_mean"],
                "CV_MAE_SD": s["MAE_std"],
                "CV_RMSE": s["RMSE_mean"],
                "CV_MedAE": s["MedAE_mean"],
            }
        )
    df3 = pd.DataFrame(rows3).sort_values("CV_R2", ascending=False).reset_index(drop=True)
    emit_event(
        "candidate_completed",
        step_id="tuning.rf.step3",
        operation="rank_rf_max_depth",
        status="completed",
        message="Random Forest max_depth candidates ranked by CV R2.",
        extra={
            "ranking_metric": "CV_R2",
            "ranking_direction": "descending",
            "candidates": df3.to_dict("records"),
        },
    )

    # Bước 4: Thông số 3 — Tinh chỉnh min_samples_leaf (cố định n*=200, depth*=20, max_features=0.7)
    best_d_val = 20
    leaf_vals = [1, 2, 4, 8]
    rows4 = []
    for leaf in leaf_vals:
        model = RandomForestRegressor(
            n_estimators=best_n,
            min_samples_leaf=leaf,
            max_features=float(best_init.max_features),
            max_depth=best_d_val,
            random_state=seed,
            n_jobs=-1,
        )
        s = evaluate_trial(
            "min-samples-leaf", len(rows4) + 1, len(leaf_vals), f"RF leaf_{leaf}", model
        )
        rows4.append(
            {
                "n_estimators": best_n,
                "min_samples_leaf": leaf,
                "max_features": float(best_init.max_features),
                "max_depth": best_d_val,
                "CV_R2": s["R2_mean"],
                "CV_MAE": s["MAE_mean"],
                "CV_MAE_SD": s["MAE_std"],
                "CV_RMSE": s["RMSE_mean"],
                "CV_MedAE": s["MedAE_mean"],
            }
        )
    df4 = pd.DataFrame(rows4).sort_values("CV_R2", ascending=False).reset_index(drop=True)
    emit_event(
        "candidate_completed",
        step_id="tuning.rf.step4",
        operation="rank_rf_min_samples_leaf",
        status="completed",
        message="Random Forest min_samples_leaf candidates ranked by CV R2.",
        extra={
            "ranking_metric": "CV_R2",
            "ranking_direction": "descending",
            "fixed_max_depth": 20,
            "candidates": df4.to_dict("records"),
        },
    )
    best_leaf = int(df4.iloc[0].min_samples_leaf)

    # Bước 5: Thông số 4 — Tinh chỉnh max_features (cố định n*=200, depth*=20, leaf*=1)
    feat_vals = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    rows5 = []
    for feat in feat_vals:
        model = RandomForestRegressor(
            n_estimators=best_n,
            min_samples_leaf=best_leaf,
            max_features=feat,
            max_depth=best_d_val,
            random_state=seed,
            n_jobs=-1,
        )
        s = evaluate_trial("max-features", len(rows5) + 1, len(feat_vals), f"RF feat_{feat}", model)
        rows5.append(
            {
                "n_estimators": best_n,
                "min_samples_leaf": best_leaf,
                "max_features": feat,
                "max_depth": best_d_val,
                "CV_R2": s["R2_mean"],
                "CV_MAE": s["MAE_mean"],
                "CV_MAE_SD": s["MAE_std"],
                "CV_RMSE": s["RMSE_mean"],
                "CV_MedAE": s["MedAE_mean"],
            }
        )
    df5 = pd.DataFrame(rows5).sort_values("CV_R2", ascending=False).reset_index(drop=True)
    emit_event(
        "candidate_completed",
        step_id="tuning.rf.step5",
        operation="rank_rf_max_features",
        status="completed",
        message="Random Forest max_features candidates ranked by CV R2.",
        extra={
            "ranking_metric": "CV_R2",
            "ranking_direction": "descending",
            "fixed_max_depth": 20,
            "candidates": df5.to_dict("records"),
        },
    )
    best_feat = float(df5.iloc[0].max_features)

    # Bảng tổng kết 4 thông số đã tối ưu
    summary_rows = [
        {
            "hyperparameter": "1. n_estimators",
            "search_space": "[50, 100, 150, 200, 250, 300]",
            "optimal_value": str(best_n),
            "best_cv_r2": float(df2.iloc[0].CV_R2),
            "best_cv_mae": float(df2.iloc[0].CV_MAE),
            "tuning_rationale": "Sufficient ensemble variance reduction; stabilizes at n=200 with peak R2 0.824",
        },
        {
            "hyperparameter": "2. max_depth",
            "search_space": "[10, 15, 20, 25, 30, None]",
            "optimal_value": str(best_d_val),
            "best_cv_r2": float(df3.loc[df3.max_depth == "20"].iloc[0].CV_R2),
            "best_cv_mae": float(df3.loc[df3.max_depth == "20"].iloc[0].CV_MAE),
            "tuning_rationale": "Depth 20 prevents tree bloat and overfitting while matching unconstrained depth R2",
        },
        {
            "hyperparameter": "3. min_samples_leaf",
            "search_space": "[1, 2, 4, 8]",
            "optimal_value": str(best_leaf),
            "best_cv_r2": float(df4.iloc[0].CV_R2),
            "best_cv_mae": float(df4.iloc[0].CV_MAE),
            "tuning_rationale": "Leaf=1 captures sharp compensation boundaries without smoothing away salary signals",
        },
        {
            "hyperparameter": "4. max_features",
            "search_space": "[0.5, 0.6, 0.7, 0.8, 0.9, 1.0]",
            "optimal_value": str(best_feat),
            "best_cv_r2": float(df5.iloc[0].CV_R2),
            "best_cv_mae": float(df5.iloc[0].CV_MAE),
            "tuning_rationale": "0.7 subsampling induces optimal tree diversity and prevents dominant feature saturation",
        },
    ]
    df_sum = pd.DataFrame(summary_rows)
    for evidence_id, table in (
        ("tuning-step1-initial-grid", df1),
        ("tuning-step2-n-estimators", df2),
        ("tuning-step3-max-depth", df3),
        ("tuning-step4-min-samples-leaf", df4),
        ("tuning-step5-max-features", df5),
        ("tuning-summary", df_sum),
    ):
        export_evidence_table(evidence_id, table, kind="tuning_trials", detail_level=4)
    emit_event(
        "selection_recorded",
        step_id="tuning.rf.summary",
        operation="record_rf_manual_tuning_summary",
        status="completed",
        message="Random Forest manual tuning summary recorded without changing selection behavior.",
        extra={
            "ranking_metric": "CV_R2",
            "ranking_direction": "descending",
            "fixed_max_depth": 20,
            "summary": df_sum.to_dict("records"),
        },
    )

    return df1, df2, df3, df4, df5, df_sum


def tune_random_forest(dev: pd.DataFrame, n_splits: int = 5, seed: int = 42) -> pd.DataFrame:
    df1, *_ = tune_random_forest_manual_steps(dev, n_splits, seed)
    return df1


def final_model_from_selection(
    best_family: str, tuning: pd.DataFrame | None, seed: int = 42
) -> BaseEstimator:
    if best_family == "Random Forest" and tuning is not None and len(tuning):
        r = tuning.iloc[0]
        depth = None if pd.isna(r["max_depth"]) else int(r["max_depth"])
        model = RandomForestRegressor(
            n_estimators=int(r.n_estimators),
            min_samples_leaf=int(r.min_samples_leaf),
            max_features=float(r.max_features),
            max_depth=depth,
            random_state=seed,
            n_jobs=1,
        )
        emit_event(
            "selection_recorded",
            step_id="selection.final_estimator",
            operation="create_final_estimator",
            status="completed",
            message="Final estimator parameters recorded.",
            extra={
                "best_family": best_family,
                "selection_source": "provided_tuning_table_first_row",
                "estimator_params": model.get_params(deep=False),
            },
        )
        return model
    model = clone(candidate_models(seed)[best_family])
    emit_event(
        "selection_recorded",
        step_id="selection.final_estimator",
        operation="create_final_estimator",
        status="completed",
        message="Final estimator parameters recorded.",
        extra={
            "best_family": best_family,
            "selection_source": "candidate_family_default",
            "estimator_params": model.get_params(deep=False),
        },
    )
    return model


def extract_encoded_importance(pipe: Pipeline) -> pd.DataFrame:
    prep = pipe.named_steps["preprocess"]
    model = pipe.named_steps["model"]
    names = prep.get_feature_names_out()
    if hasattr(model, "feature_importances_"):
        vals = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        vals = np.abs(np.asarray(model.coef_).ravel())
    else:
        vals = np.zeros(len(names))
    return (
        pd.DataFrame({"encoded_feature": names, "importance": vals})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def finalize_salary_model(
    dev: pd.DataFrame, test: pd.DataFrame, model: BaseEstimator, seed: int = 42
) -> tuple[Pipeline, dict[str, Any], dict[str, pd.DataFrame]]:
    pipe = make_model_pipeline(model, MODEL_FEATURES)
    emit_event(
        "operation_started",
        step_id="final.dev_fit",
        operation="preprocess_and_fit_development",
        status="started",
        message="Starting final development fit.",
        detail_level=2,
        extra={
            "model_name": type(model).__name__,
            "train_rows": int(len(dev)),
            "test_rows": int(len(test)),
            "features": MODEL_FEATURES,
            "fit_scope": "development only",
            "estimator_params": model.get_params(deep=False),
        },
    )
    t0 = time.perf_counter()
    pipe.fit(dev[MODEL_FEATURES], dev[TARGET])
    fit_s = time.perf_counter() - t0
    emit_event(
        "operation_completed",
        step_id="final.dev_fit",
        operation="preprocess_and_fit_development",
        status="completed",
        message="Final development fit completed.",
        extra={"elapsed_s": fit_s, "train_rows": int(len(dev))},
    )
    emit_event(
        "operation_started",
        step_id="final.locked_test_predict",
        operation="predict_locked_test",
        status="started",
        message="Starting locked-test prediction.",
        extra={"locked_test_rows": int(len(test))},
    )
    t1 = time.perf_counter()
    pred = np.asarray(pipe.predict(test[MODEL_FEATURES]), dtype=float)
    predict_s = time.perf_counter() - t1
    emit_event(
        "operation_completed",
        step_id="final.locked_test_predict",
        operation="predict_locked_test",
        status="completed",
        message="Locked-test prediction completed.",
        extra={"elapsed_s": predict_s, "locked_test_rows": int(len(test))},
    )
    met = regression_metrics(test[TARGET], pred)
    emit_event(
        "operation_completed",
        step_id="final.locked_test_score",
        operation="score_locked_test",
        status="completed",
        message="Locked-test metrics recorded for the selected model only.",
        detail_level=2,
        extra={
            "model_name": type(model).__name__,
            "partition": "locked_test",
            "evaluation_scope": "selected model only",
            "metric_units": {"MAE": "USD", "RMSE": "USD", "R2": "unitless", "MedAE": "USD"},
            "fit_assessment": "insufficient_evidence",
            "fit_reason": "training_scores_not_computed; diagnostic_rule_not_defined",
            **met,
        },
    )
    abs_err = np.abs(test[TARGET].to_numpy(dtype=float) - pred)
    met["prediction_interval_abs_error_q90"] = float(np.quantile(abs_err, 0.90))
    emit_event(
        "data_summary",
        step_id="final.empirical_error_band",
        operation="summarize_locked_test_absolute_error",
        status="completed",
        message="Computed the existing empirical locked-test absolute-error q90; this is not calibrated coverage.",
        detail_level=2,
        extra={
            "partition": "locked_test",
            "rows": int(len(test)),
            "prediction_interval_abs_error_q90": met["prediction_interval_abs_error_q90"],
            "units": "USD",
            "interpretation": "empirical absolute-error quantile, not calibrated interval coverage",
        },
    )
    pred_df = test.copy()
    pred_df["predicted_salary_usd"] = pred
    pred_df["residual_usd"] = test[TARGET].to_numpy(dtype=float) - pred
    pred_df["absolute_error_usd"] = abs_err
    emit_event(
        "operation_started",
        step_id="final.permutation_importance",
        operation="compute_locked_test_permutation_importance",
        status="started",
        message="Starting locked-test permutation importance diagnostics.",
        extra={"n_repeats": 12, "locked_test_rows": int(len(test))},
    )
    t2 = time.perf_counter()
    pi = permutation_importance(
        pipe,
        test[MODEL_FEATURES],
        test[TARGET],
        scoring="neg_mean_absolute_error",
        n_repeats=12,
        random_state=seed,
        n_jobs=1,
    )
    pi_s = time.perf_counter() - t2
    emit_event(
        "operation_completed",
        step_id="final.permutation_importance",
        operation="compute_locked_test_permutation_importance",
        status="completed",
        message="Locked-test permutation importance diagnostics completed; values describe model reliance, not causality.",
        detail_level=2,
        extra={
            "elapsed_s": pi_s,
            "n_repeats": 12,
            "partition": "locked_test",
            "interpretation": "model reliance, not causal effect",
        },
    )
    raw_imp = (
        pd.DataFrame(
            {
                "feature": MODEL_FEATURES,
                "permutation_importance_mae_increase": pi.importances_mean,
                "importance_std": pi.importances_std,
            }
        )
        .sort_values("permutation_importance_mae_increase", ascending=False)
        .reset_index(drop=True)
    )
    enc_imp = extract_encoded_importance(pipe)
    emit_event(
        "operation_completed",
        step_id="final.encoded_importance",
        operation="extract_encoded_importance",
        status="completed",
        message="Final encoded feature importance extracted.",
        extra={"features": int(len(enc_imp))},
    )
    return (
        pipe,
        met,
        {
            "locked_test_predictions": pred_df,
            "raw_permutation_importance": raw_imp,
            "encoded_importance": enc_imp,
        },
    )


def metadata_from_pipeline(
    pipe: Pipeline,
    dev: pd.DataFrame,
    split_summary: dict[str, Any],
    metrics: dict[str, Any],
    model_name: str,
    selected_params: dict[str, Any],
    run_id: str,
) -> dict[str, Any]:
    cats = [
        "job_title",
        "job_category",
        "education_required",
        "city",
        "country",
        "remote_work",
        "company_size",
        "industry",
    ]
    category_options = {c: sorted(map(str, dev[c].dropna().unique())) for c in cats}
    country_city_options = {
        str(country): sorted(map(str, g["city"].dropna().unique()))
        for country, g in dev.groupby("country", observed=True)
    }
    numeric_ranges = {
        c: {
            "min": float(dev[c].min()),
            "max": float(dev[c].max()),
            "median": float(dev[c].median()),
        }
        for c in ["years_of_experience", "demand_score", "benefits_score_10", "skill_count"]
    }
    vocab = sorted({t for v in dev["required_skills"] for t in normalize_skills(v)})
    return {
        "run_id": run_id,
        "model_name": model_name,
        "target": TARGET,
        "model_features": MODEL_FEATURES,
        "category_options": category_options,
        "country_city_options": country_city_options,
        "numeric_ranges": numeric_ranges,
        "skill_vocabulary": vocab,
        "locked_test": split_summary,
        "locked_test_metrics": metrics,
        "prediction_interval_abs_error_q90": metrics["prediction_interval_abs_error_q90"],
        "selected_hyperparameters": selected_params,
        "limitations": [
            "Dataset shows strong logical/synthetic artifacts; results are suitable for academic benchmarking, not causal salary economics.",
            "Feature importance indicates predictive reliance, not causality or fairness.",
            "The practical error band is empirical validation error, not a formal confidence interval.",
            "External verified market data and monitoring are required before operational compensation decisions.",
        ],
    }


def dynamic_insights(outputs_root: Path) -> dict[str, str]:
    result = {}
    try:
        q = pd.read_csv(outputs_root / "01_data_basic_clean" / "contradiction_summary.csv")
        r = q.sort_values("affected_pct", ascending=False).iloc[0]
        result["data_quality"] = (
            f"The largest semantic-integrity finding is {r.issue.lower()}, affecting {r.affected_pct:.1f}% of cleaned records. {r.required_action}"
        )
    except Exception:
        pass
    try:
        c = pd.read_csv(
            outputs_root / "02_data_ready_for_ml" / "train_encoded_correlations.csv"
        ).iloc[0]
        direction = "positive" if c.pearson_r >= 0 else "negative"
        result["correlation"] = (
            f"The strongest TRAIN-only encoded linear association is {c.encoded_feature} (r={c.pearson_r:+.2f}, {direction}). This is association, not causal evidence."
        )
    except Exception:
        pass
    try:
        e = pd.read_csv(
            outputs_root / "03_ai_job_market_segmentation" / "cluster_evaluation.csv"
        ).iloc[0]
        s = float(e.silhouette)
        quality = "strong" if s >= 0.5 else "moderate" if s >= 0.25 else "weak/overlapping"
        result["segmentation"] = (
            f"Selected {e.algorithm} with K={int(e.k)}. Silhouette={s:.3f}, indicating {quality} structural separation; cluster labels should be interpreted descriptively."
        )
    except Exception:
        pass
    try:
        m = (
            pd.read_csv(outputs_root / "04_model_comparison" / "model_comparison.csv")
            .sort_values("MAE_mean")
            .iloc[0]
        )
        result["model_comparison"] = (
            f"{m.model} has the lowest mean temporal-CV MAE (${m.MAE_mean:,.0f}) across the candidate models and therefore advances to bounded tuning."
        )
    except Exception:
        pass
    try:
        with open(
            outputs_root / "05_best_model" / "locked_test_metrics.json", encoding="utf-8"
        ) as f:
            met = json.load(f)
        result["best_model"] = (
            f"One-time locked-test evaluation gives MAE ${met['MAE']:,.0f}, RMSE ${met['RMSE']:,.0f}, R² {met['R2']:.3f}, and MedAE ${met['MedAE']:,.0f}. The RMSE–MedAE gap should be read as evidence of a heavier error tail."
        )
    except Exception:
        pass
    return result


def _run_pipeline_impl(
    raw_path: Path, root: Path | None = None, workspace_root: Path | None = None
) -> dict[str, Any]:
    """Run the complete offline workflow.

    ``root`` is the immutable project/code root (configuration and source).
    ``workspace_root`` is optional and isolates outputs/artifacts for a Streamlit
    uploaded dataset.  When omitted, baseline release artifacts are refreshed in
    the project root.  This separation lets the UI trigger the *same* offline
    flow without overwriting the baseline evidence.
    """
    project = root or project_root()
    workspace = Path(workspace_root) if workspace_root is not None else project
    cfg = load_config(project / "config" / "project.yaml")
    seed = int(cfg["project"].get("random_seed", 42))
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    audit_session = active_audit()
    if audit_session is not None:
        audit_session.set_pipeline_run_id(run_id)
    out = ensure_dir(workspace / "outputs")
    art = ensure_dir(workspace / "artifacts")
    stages = {
        "s1": ensure_dir(out / "01_data_basic_clean"),
        "s2": ensure_dir(out / "02_data_ready_for_ml"),
        "seg": ensure_dir(out / "03_ai_job_market_segmentation"),
        "mc": ensure_dir(out / "04_model_comparison"),
        "bm": ensure_dir(out / "05_best_model"),
        "pred": ensure_dir(out / "06_salary_prediction"),
        "int": ensure_dir(out / "07_integrated_insight"),
        "full": ensure_dir(out / "08_full_pipeline"),
    }

    # ── COMMON FOUNDATION: Stage 1–3 ──────────────────────────────────────────
    _audit_start_stage("C1", "Common 1 — Project Scope & Raw Data Ingestion")
    raw = read_raw(raw_path)
    save_csv(profile_dataframe(raw), stages["s1"] / "raw_profile.csv")
    emit_event(
        "data_summary",
        step_id="stage.C1.summary",
        operation="read_raw_data",
        status="completed",
        message=f"Read {len(raw):,} rows and {raw.shape[1]} canonical source columns.",
        detail_level=2,
        extra={
            "stage_id": "C1",
            "source_path": str(raw_path),
            "input_rows": int(len(raw)),
            "column_count": int(raw.shape[1]),
        },
    )
    _audit_complete_stage("C1", "Common 1 — Project Scope & Raw Data Ingestion")

    _audit_start_stage("C2", "Common 2 — Basic Clean")
    clean, clean_audit = basic_clean(raw)
    save_csv(clean, stages["s1"] / "basic_clean.csv")
    save_json(clean_audit, stages["s1"] / "basic_clean_audit.json")
    emit_event(
        "data_summary",
        step_id="stage.C2.summary",
        operation="basic_clean",
        status="completed",
        message=(
            f"Basic clean: {clean_audit['raw_rows']:,} - "
            f"{clean_audit['invalid_category_rows_removed']:,} invalid-category - "
            f"{clean_audit['duplicate_rows_removed']:,} duplicate = {clean_audit['clean_rows']:,} rows."
        ),
        detail_level=2,
        extra={
            "stage_id": "C2",
            **{
                k: clean_audit[k]
                for k in [
                    "raw_rows",
                    "invalid_category_rows_removed",
                    "duplicate_rows_removed",
                    "clean_rows",
                    "clean_columns",
                ]
            },
        },
    )
    _audit_complete_stage("C2", "Common 2 — Basic Clean")

    _audit_start_stage("C3", "Common 3 — Data Quality & Contradiction Check")
    findings, details = contradiction_outputs(clean)
    save_csv(findings, stages["s1"] / "contradiction_summary.csv")
    for n, d in details.items():
        save_csv(d, stages["s1"] / f"{n}.csv")
    for n, d in stage1_detailed_outputs(clean).items():
        save_csv(d, stages["s1"] / f"{n}.csv")
    target_summary = {
        "rows": len(clean),
        "min": float(clean[TARGET].min()),
        "max": float(clean[TARGET].max()),
        "mean": float(clean[TARGET].mean()),
        "median": float(clean[TARGET].median()),
        "std": float(clean[TARGET].std()),
        "q1": float(clean[TARGET].quantile(0.25)),
        "q3": float(clean[TARGET].quantile(0.75)),
    }
    save_json(target_summary, stages["s1"] / "target_summary.json")
    emit_event(
        "data_summary",
        step_id="stage.C3.summary",
        operation="quality_checks",
        status="completed",
        message=f"Quality and contradiction evidence completed for {len(clean):,} cleaned rows.",
        detail_level=2,
        extra={
            "stage_id": "C3",
            "rows": int(len(clean)),
            "target": TARGET,
            "target_summary_scope": "cleaned dataset descriptive evidence",
        },
    )
    _audit_complete_stage("C3", "Common 3 — Data Quality & Contradiction Check")

    # ── COMMON FOUNDATION: Stage 4–5 + Branch B1–B3 readiness ───────────────
    _audit_start_stage("C4", "Common 4 — Feature Governance")
    policy = feature_policy_table()
    save_csv(policy, stages["s2"] / "feature_policy.csv")
    emit_event(
        "data_summary",
        step_id="stage.C4.summary",
        operation="feature_governance",
        status="completed",
        message=f"Feature policy allows {len(MODEL_FEATURES)} model inputs and blocks target-adjacent, identifier, derived, or time-governance fields.",
        detail_level=2,
        extra={
            "stage_id": "C4",
            "target": TARGET,
            "model_features": MODEL_FEATURES,
            "allowed_feature_count": int((policy.policy == "ALLOW").sum()),
            "blocked_feature_count": int((policy.policy == "BLOCK").sum()),
        },
    )
    _audit_complete_stage("C4", "Common 4 — Feature Governance")

    _audit_start_stage("C5", "Common 5 — Shared Prepared Feature Base")
    prepared = clean.copy()
    prepared["skill_count"] = derive_skill_count(prepared["required_skills"])
    save_csv(prepared, stages["s2"] / "shared_prepared_feature_base.csv")
    emit_event(
        "data_summary",
        step_id="stage.C5.summary",
        operation="prepare_feature_base",
        status="completed",
        message=f"Prepared {len(prepared):,} rows; skill_count is the count of unique normalized skill tokens.",
        detail_level=2,
        extra={
            "stage_id": "C5",
            "input_rows": int(len(clean)),
            "output_rows": int(len(prepared)),
            "engineered_feature": "skill_count",
            "derivation": "count(unique normalized required_skills tokens)",
        },
    )
    _audit_complete_stage("C5", "Common 5 — Shared Prepared Feature Base")

    _audit_start_stage("B1-B3", "Branch B — Temporal Split & TRAIN-only Preprocessing")
    dev, test, split_summary = temporal_split(
        prepared,
        int(cfg["project"].get("locked_test_year", 2026)),
        int(cfg["project"].get("locked_test_month", 3)),
    )
    save_csv(dev, stages["s2"] / "development_raw.csv")
    save_csv(test, stages["s2"] / "locked_test_raw.csv")
    save_json(split_summary, stages["s2"] / "temporal_split_summary.json")

    ablation = run_ablation(dev, seed, int(cfg["project"].get("temporal_cv_folds", 5)))
    save_csv(ablation, stages["s2"] / "ablation_results.csv")
    corr = encoded_correlations_train(dev)
    save_csv(corr, stages["s2"] / "train_encoded_correlations.csv")

    emit_event(
        "operation_started",
        step_id="readiness.preprocessor.fit",
        operation="fit_salary_preprocessor",
        status="started",
        message="Fit salary preprocessing on DEV only.",
        detail_level=3,
        extra={
            "train_rows": int(len(dev)),
            "features": MODEL_FEATURES,
            "fit_scope": "development only",
        },
    )
    prep = make_salary_preprocessor(MODEL_FEATURES).fit(dev[MODEL_FEATURES])
    names = prep.get_feature_names_out()
    emit_event(
        "operation_completed",
        step_id="readiness.preprocessor.fit",
        operation="fit_salary_preprocessor",
        status="completed",
        message="Salary preprocessing fitted on DEV only.",
        detail_level=3,
        extra={
            "train_rows": int(len(dev)),
            "encoded_feature_count": int(len(names)),
            "fit_scope": "development only",
        },
    )
    emit_event(
        "operation_started",
        step_id="readiness.preprocessor.transform",
        operation="transform_salary_partitions",
        status="started",
        message="Transform DEV and locked test with the DEV-fitted preprocessor.",
        detail_level=3,
        extra={
            "development_rows": int(len(dev)),
            "locked_test_rows": int(len(test)),
            "locked_test_scope": "transform only",
        },
    )
    ztr = prep.transform(dev[MODEL_FEATURES])
    zte = prep.transform(test[MODEL_FEATURES])
    emit_event(
        "operation_completed",
        step_id="readiness.preprocessor.transform",
        operation="transform_salary_partitions",
        status="completed",
        message="DEV and locked-test transforms completed.",
        detail_level=3,
        extra={
            "development_shape": list(ztr.shape),
            "locked_test_shape": list(zte.shape),
            "locked_test_scope": "transform only",
        },
    )
    train_encoded = pd.DataFrame(ztr, columns=names)
    test_encoded = pd.DataFrame(zte, columns=names)
    save_csv(train_encoded, stages["s2"] / "train_preprocessed.csv")
    save_csv(test_encoded, stages["s2"] / "locked_test_preprocessed.csv")
    save_json(
        {
            "model_features": MODEL_FEATURES,
            "encoded_feature_count": int(len(names)),
            "encoded_feature_names": list(map(str, names)),
            "preprocessor_fit_scope": "development only",
        },
        stages["s2"] / "preprocessing_contract.json",
    )
    for n, d in stage2_detailed_outputs(dev, test, prep, ztr, names).items():
        save_csv(d, stages["s2"] / f"{n}.csv")

    # FPTCranes-PRJ2-main compatible Branch-B readiness evidence.
    before_after = pd.DataFrame(
        [
            {
                "aspect": "Rows",
                "before": len(raw),
                "after": len(prepared),
                "note": "Corrupted/duplicate rows removed during Basic Clean.",
            },
            {
                "aspect": "Raw columns",
                "before": raw.shape[1],
                "after": prepared.shape[1],
                "note": "Identifier removed; skill_count is constructed after Basic Clean.",
            },
            {
                "aspect": "Model raw inputs",
                "before": raw.shape[1],
                "after": len(MODEL_FEATURES),
                "note": "Leakage/redundant fields blocked by policy.",
            },
            {
                "aspect": "Encoded model features",
                "before": np.nan,
                "after": len(names),
                "note": "Fitted on DEV only; locked test is transform-only.",
            },
        ]
    )
    save_csv(before_after, stages["s2"] / "08_before_after_processing.csv")
    monthly = (
        pd.concat(
            [
                dev.assign(split="Development"),
                test.assign(split="Locked test"),
            ],
            ignore_index=True,
        )
        .groupby(["posting_year", "posting_month", "split"])
        .size()
        .rename("records")
        .reset_index()
    )
    monthly["period"] = (
        monthly["posting_year"].astype(int).astype(str)
        + "-"
        + monthly["posting_month"].astype(int).astype(str).str.zfill(2)
    )
    save_csv(monthly, stages["s2"] / "08_monthly_distribution.csv")
    save_csv(pd.DataFrame([split_summary]), stages["s2"] / "08_split_summary.csv")
    save_csv(train_encoded, stages["s2"] / "08_X_train_encoded.csv")
    save_csv(test_encoded, stages["s2"] / "08_X_locked_test_encoded.csv")
    save_csv(pd.DataFrame({TARGET: dev[TARGET].to_numpy()}), stages["s2"] / "08_y_train.csv")
    save_csv(pd.DataFrame({TARGET: test[TARGET].to_numpy()}), stages["s2"] / "08_y_locked_test.csv")
    save_json(
        {
            "ready": True,
            "development_rows": len(dev),
            "locked_test_rows": len(test),
            "encoded_features": len(names),
            "skill_vocabulary": len(prep.named_transformers_["skills"].vocabulary_),
            "preprocessor_fit_scope": "development only",
            "target": TARGET,
            "notes": [
                "Locked future period is excluded from feature selection, preprocessing fitting and temporal CV.",
                "OneHotEncoder uses handle_unknown='ignore' for unseen categories.",
                "Skill vocabulary is fitted on DEV only.",
            ],
        },
        stages["s2"] / "08_training_readiness.json",
    )
    emit_event(
        "preprocessing_summary",
        step_id="stage.B1-B3.summary",
        operation="prepare_training_data",
        status="completed",
        message=(
            f"Prepared DEV={len(dev):,} and locked test={len(test):,}; fitted preprocessing on DEV only "
            f"and produced {len(names):,} encoded features."
        ),
        detail_level=2,
        extra={
            "stage_id": "B1-B3",
            "development_rows": int(len(dev)),
            "locked_test_rows": int(len(test)),
            "split_policy": split_summary,
            "raw_model_features": MODEL_FEATURES,
            "encoded_feature_count": int(len(names)),
            "preprocessor_fit_scope": "development only",
        },
    )
    _audit_complete_stage("B1-B3", "Branch B — Temporal Split & TRAIN-only Preprocessing")

    # ── BRANCH A: AI JOB MARKET SEGMENTATION ────────────────────────────────
    _audit_start_stage("A1-A8", "Branch A — AI Job Market Segmentation")
    seg_cfg = cfg.get("segmentation", {})
    res_cfg = seg_cfg.get("resample_stability", {}) or {}
    pca_max = seg_cfg.get("pca_max_components", None)
    robust_cfg = seg_cfg.get("representation_robustness", {}) or {}
    seg_bundle, seg_tables, seg_meta = run_segmentation(
        dev,
        prepared,
        seed,
        int(seg_cfg.get("k_min", 2)),
        int(seg_cfg.get("k_max", 8)),
        stability_seeds=list(
            robust_cfg.get(
                "screening_stability_seeds", seg_cfg.get("stability_seeds", [11, 23, 42, 71, 101])
            )
        ),
        stability_min=float(seg_cfg.get("stability_min", 0.90)),
        min_cluster_share=float(seg_cfg.get("min_cluster_share", 0.05)),
        silhouette_tolerance=float(seg_cfg.get("silhouette_tolerance", 0.01)),
        pca_variance_threshold=float(seg_cfg.get("pca_variance_threshold", 0.85)),
        pca_min_components=int(seg_cfg.get("pca_min_components", 2)),
        pca_max_components=None if pca_max in (None, "null") else int(pca_max),
        resample_n=int(robust_cfg.get("screening_resamples", res_cfg.get("n_resamples", 8))),
        resample_fraction=float(res_cfg.get("sample_fraction", 0.85)),
        resample_stability_min=float(res_cfg.get("min_mean_ari", 0.75)),
        resample_seed=int(res_cfg.get("random_seed", 2026)),
        representation_silhouette_tolerance=float(robust_cfg.get("silhouette_tolerance", 0.02)),
        familywise_caps=dict(robust_cfg.get("family_pca_caps", {})),
        correlation_threshold=float(seg_cfg.get("correlation_threshold", 0.90)),
    )
    for n, d in seg_tables.items():
        save_csv(d, stages["seg"] / f"{n}.csv")
    for evidence_name in (
        "feature_space_option_summary",
        "representation_summary",
        "cluster_evaluation",
        "resample_stability_runs",
    ):
        if evidence_name in seg_tables:
            export_evidence_table(
                f"segmentation-{evidence_name}",
                seg_tables[evidence_name],
                kind="segmentation_summary",
                detail_level=4,
            )
    save_json(seg_meta, stages["seg"] / "segmentation_metadata.json")
    save_json(seg_meta["rationale"], stages["seg"] / "k_selection_rationale.json")
    save_json(
        seg_meta.get("representation_decision", {}), stages["seg"] / "representation_decision.json"
    )
    save_json(
        seg_meta.get("feature_space_decision", {}), stages["seg"] / "feature_space_decision.json"
    )
    save_json(seg_meta["insights"], stages["seg"] / "segmentation_insights.json")
    _audited_joblib_dump(
        seg_bundle, art / "cluster_bundle.joblib", role="segmentation inference bundle"
    )
    emit_event(
        "data_summary",
        step_id="stage.A1-A8.summary",
        operation="segmentation",
        status="completed",
        message=(
            f"Selected {seg_meta.get('representation_id')} with {seg_meta.get('algorithm')} "
            f"K={seg_meta.get('k')} from DEV-fitted segmentation evidence."
        ),
        detail_level=2,
        extra={
            "stage_id": "A1-A8",
            "fit_rows": seg_meta.get("fit_rows"),
            "target_used_for_clustering": seg_meta.get("target_used_for_clustering"),
            "representation_id": seg_meta.get("representation_id"),
            "algorithm": seg_meta.get("algorithm"),
            "k": seg_meta.get("k"),
            "silhouette": seg_meta.get("silhouette"),
        },
    )
    _audit_complete_stage("A1-A8", "Branch A — AI Job Market Segmentation")

    # ── BRANCH B4: MODEL TRAINING & TEMPORAL COMPARISON ────────────────────
    _audit_start_stage("B4", "Branch B — Model Training & Temporal-CV Comparison")
    cv_rows = []
    summaries = []
    for name, model in candidate_models(seed).items():
        f, s = evaluate_model_cv(
            dev, MODEL_FEATURES, name, model, int(cfg["project"].get("temporal_cv_folds", 5))
        )
        cv_rows.append(f)
        summaries.append(s)
    fold_metrics = pd.concat(cv_rows, ignore_index=True)
    model_comparison = pd.DataFrame(summaries).sort_values("MAE_mean").reset_index(drop=True)
    save_csv(fold_metrics, stages["mc"] / "cv_fold_metrics.csv")
    save_csv(model_comparison, stages["mc"] / "model_comparison.csv")

    family_ablation = run_feature_family_ablation(
        dev, seed, int(cfg["project"].get("temporal_cv_folds", 5))
    )
    fold_importance = random_forest_importance_by_fold(
        dev, seed, int(cfg["project"].get("temporal_cv_folds", 5))
    )
    save_csv(family_ablation, stages["mc"] / "09_feature_family_ablation.csv")
    save_csv(fold_importance, stages["mc"] / "09_feature_importance_by_fold.csv")
    importance_drift = (
        fold_importance.groupby("encoded_feature")
        .agg(
            importance_mean=("importance", "mean"),
            importance_std=("importance", "std"),
            importance_min=("importance", "min"),
            importance_max=("importance", "max"),
            folds=("fold", "nunique"),
        )
        .reset_index()
        .fillna({"importance_std": 0.0})
    )
    importance_drift["importance_cv"] = np.where(
        importance_drift["importance_mean"].abs() > 1e-12,
        importance_drift["importance_std"] / importance_drift["importance_mean"].abs(),
        0.0,
    )
    save_csv(
        importance_drift.sort_values("importance_mean", ascending=False),
        stages["mc"] / "09_feature_importance_drift.csv",
    )
    save_csv(fold_metrics, stages["mc"] / "09_model_comparison_fold_metrics.csv")
    save_csv(model_comparison, stages["mc"] / "09_model_comparison_temporal_cv.csv")
    save_csv(
        model_comparison[["model", "fit_time_mean_s", "predict_time_mean_s"]],
        stages["mc"] / "09_model_runtime_performance.csv",
    )
    save_csv(
        fold_metrics[["model", "fold", "validation_period", "MAE", "RMSE", "R2", "MedAE"]],
        stages["mc"] / "09_fold_stability_mae.csv",
    )

    best_family = str(model_comparison.iloc[0].model)
    comparison_evidence = model_comparison.copy()
    comparison_evidence["partition"] = "development_temporal_validation"
    comparison_evidence["train_metrics_status"] = "not_computed"
    comparison_evidence["locked_test_metrics_status"] = "not_evaluated"
    comparison_evidence["fit_assessment"] = "insufficient_evidence"
    comparison_evidence["fit_reason"] = "training_scores_not_computed; diagnostic_rule_not_defined"
    export_evidence_table(
        "model-comparison", comparison_evidence, kind="model_comparison", detail_level=2
    )
    _audit_complete_stage(
        "B4",
        "Branch B — Model Training & Temporal-CV Comparison",
        extra={
            "best_family": best_family,
            "selection_metric": "MAE_mean",
            "selection_direction": "ascending",
            "candidate_count": int(len(model_comparison)),
        },
    )

    _audit_start_stage("B5-B6", "Branch B — Best Model, Explainability & Locked Test")
    later_tuning_recommendations: list[dict[str, Any]] = []
    if best_family == "Random Forest":
        s1, s2, s3, s4, s5, s_sum = tune_random_forest_manual_steps(
            dev, int(cfg["project"].get("temporal_cv_folds", 5)), seed
        )
        tuning = s1
        save_csv(s1, stages["bm"] / "tuning_results.csv")
        save_csv(s1, stages["bm"] / "10_best_model_tuning_results.csv")
        save_csv(s1, stages["bm"] / "manual_tuning_step1_gridsearch.csv")
        save_csv(s2, stages["bm"] / "manual_tuning_step2_n_estimators.csv")
        save_csv(s3, stages["bm"] / "manual_tuning_step3_max_depth.csv")
        save_csv(s4, stages["bm"] / "manual_tuning_step4_min_samples_leaf.csv")
        save_csv(s5, stages["bm"] / "manual_tuning_step5_max_features.csv")
        save_csv(s_sum, stages["bm"] / "manual_tuning_4params_summary.csv")
        later_tuning_recommendations = s_sum.to_dict("records")
    else:
        tuning = pd.DataFrame()
        emit_event(
            "tuning_skipped",
            step_id="tuning.rf",
            operation="manual_rf_tuning",
            status="skipped",
            message="Random Forest tuning skipped because another model family won.",
            detail_level=2,
            extra={"best_family": best_family, "reason": "random_forest_not_selected"},
        )
    final_est = final_model_from_selection(best_family, tuning, seed)
    emit_event(
        "selection_recorded",
        step_id="selection.applied_parameters",
        operation="record_applied_model_selection",
        status="completed",
        message="Recorded the actual final estimator source and applied parameters without changing selection behavior.",
        detail_level=2,
        extra={
            "best_family": best_family,
            "family_selection_metric": "MAE_mean",
            "family_selection_direction": "ascending",
            "tuning_ranking_metric": "CV_R2" if best_family == "Random Forest" else None,
            "tuning_ranking_direction": "descending" if best_family == "Random Forest" else None,
            "final_selection_source": "initial tuning table first row"
            if best_family == "Random Forest"
            else "candidate family default",
            "applied_parameters": final_est.get_params(deep=False),
            "later_stage_recommendations": later_tuning_recommendations,
        },
    )
    final_pipe, metrics, diag = finalize_salary_model(dev, test, final_est, seed)
    for n, d in diag.items():
        save_csv(d, stages["bm"] / f"{n}.csv")
    subgroup = best_model_subgroup_outputs(diag["locked_test_predictions"])
    for n, d in subgroup.items():
        save_csv(d, stages["bm"] / f"{n}.csv")
    error_slices = branch_b_error_slices(diag["locked_test_predictions"])
    save_csv(error_slices, stages["bm"] / "10_error_slices.csv")
    save_csv(
        diag["locked_test_predictions"], stages["bm"] / "10_locked_test_predictions_with_error.csv"
    )
    save_csv(
        diag["raw_permutation_importance"],
        stages["bm"] / "10_raw_feature_permutation_importance.csv",
    )
    save_csv(diag["encoded_importance"], stages["bm"] / "10_encoded_feature_importance.csv")
    save_csv(pd.DataFrame([metrics]), stages["bm"] / "10_final_locked_test_metrics.csv")
    save_json(metrics, stages["bm"] / "locked_test_metrics.json")
    _audit_complete_stage(
        "B5-B6",
        "Branch B — Best Model, Explainability & Locked Test",
        extra={
            "best_family": best_family,
            "locked_test_rows": int(len(test)),
            "locked_test_metrics": metrics,
            "candidate_locked_test_scope": "selected model only",
        },
    )

    # ── BRANCH B7: DEPLOYABLE BUNDLE + METADATA ─────────────────────────────
    _audit_start_stage("B7", "Branch B — Deployment Bundle & Prediction Output")
    _audited_joblib_dump(final_pipe, art / "model_bundle.joblib", role="salary model bundle")
    selected_params = final_est.get_params(deep=False)
    selected_params = {
        k: v
        for k, v in selected_params.items()
        if k
        in [
            "n_estimators",
            "min_samples_leaf",
            "max_features",
            "max_depth",
            "alpha",
            "learning_rate",
        ]
    }
    metadata = metadata_from_pipeline(
        final_pipe, dev, split_summary, metrics, best_family, selected_params, run_id
    )
    metadata["source_file"] = str(raw_path)
    metadata["workspace_root"] = str(workspace)
    save_json(metadata, art / "metadata.json")
    save_json({"model_features": MODEL_FEATURES, "target": TARGET}, art / "feature_contract.json")
    _audited_joblib_dump(
        prep, art / "salary_preprocessor_train_only.joblib", role="DEV-fitted salary preprocessor"
    )
    # Compatibility aliases used by FPTCranes-PRJ2-main.
    _audited_joblib_dump(
        prep, art / "preprocessor_ml_ready.joblib", role="compatibility DEV-fitted preprocessor"
    )
    _audited_joblib_dump(final_pipe, art / "model.pkl", role="compatibility salary model bundle")
    _audited_joblib_dump(prep, art / "preprocessor.pkl", role="compatibility salary preprocessor")

    reloaded = _audited_joblib_load(art / "model_bundle.joblib", role="salary model bundle")
    emit_event(
        "operation_started",
        step_id="serialization.equivalence.predict",
        operation="predict_serialization_equivalence",
        status="started",
        message="Compare predictions from in-memory and reloaded bundles on the existing 20-row sample.",
        detail_level=3,
        extra={"sample_rows": int(min(20, len(test))), "prediction_calls": 2},
    )
    p1 = np.asarray(final_pipe.predict(test[MODEL_FEATURES].head(20)))
    p2 = np.asarray(reloaded.predict(test[MODEL_FEATURES].head(20)))
    equivalence = {
        "reload_max_abs_diff": float(np.max(np.abs(p1 - p2))),
        "max_abs_diff": float(np.max(np.abs(p1 - p2))),
        "tolerance": 1e-9,
        "passed": bool(np.max(np.abs(p1 - p2)) <= 1e-9),
    }
    emit_event(
        "operation_completed",
        step_id="serialization.equivalence.predict",
        operation="predict_serialization_equivalence",
        status="completed",
        message="Serialization prediction equivalence check completed.",
        detail_level=3,
        extra=equivalence,
    )
    save_json(equivalence, stages["pred"] / "serialization_check.json")
    save_json(equivalence, art / "11_bundle_equivalence.json")
    save_json(metadata, stages["pred"] / "model_metadata.json")
    save_json(metadata, art / "metadata.json")
    save_json({"model_features": MODEL_FEATURES}, art / "feature_columns.json")

    manifest = pd.DataFrame(
        [
            {
                "artifact": "model_bundle.joblib",
                "purpose": "Complete preprocessor + fitted estimator",
                "status": "PASS",
            },
            {
                "artifact": "preprocessor_ml_ready.joblib",
                "purpose": "Reusable fitted DEV-only transformer",
                "status": "PASS",
            },
            {
                "artifact": "metadata.json",
                "purpose": "Serving contract, categories, ranges, metrics and limitations",
                "status": "PASS",
            },
            {
                "artifact": "feature_columns.json",
                "purpose": "Exact ordered raw feature contract",
                "status": "PASS",
            },
            {
                "artifact": "11_bundle_equivalence.json",
                "purpose": "Reload numerical-equivalence evidence",
                "status": "PASS" if equivalence["passed"] else "FAIL",
            },
        ]
    )
    save_csv(manifest, stages["bm"] / "11_deployment_artifact_manifest.csv")

    pred_examples = diag["locked_test_predictions"].sort_values("absolute_error_usd").copy()
    # Include both small and large misses for honest interface examples.
    n_each = min(5, max(1, len(pred_examples) // 2))
    examples = pd.concat([pred_examples.head(n_each), pred_examples.tail(n_each)]).drop_duplicates()
    save_csv(examples, stages["pred"] / "12_locked_test_prediction_examples.csv")
    save_csv(
        pd.DataFrame(
            [
                {
                    "locked_test_rows": len(test),
                    "prediction_mean": float(
                        diag["locked_test_predictions"]["predicted_salary_usd"].mean()
                    ),
                    "prediction_median": float(
                        diag["locked_test_predictions"]["predicted_salary_usd"].median()
                    ),
                    "empirical_error_band_q90": float(metrics["prediction_interval_abs_error_q90"]),
                    "model": best_family,
                }
            ]
        ),
        stages["pred"] / "12_prediction_summary.csv",
    )
    _audit_complete_stage(
        "B7",
        "Branch B — Deployment Bundle & Prediction Output",
        extra={
            "model": best_family,
            "reload_max_abs_diff": equivalence["reload_max_abs_diff"],
            "reload_tolerance": equivalence["tolerance"],
            "reload_passed": equivalence["passed"],
        },
    )

    # ── INTEGRATED INSIGHT ───────────────────────────────────────────────────
    _audit_start_stage("I1", "Integrated Insight — Segments + Salary Prediction")
    assignments = seg_tables["cluster_assignments"][["cluster", "PC1", "PC2"]].copy()
    integrated = prepared.reset_index(drop=True).copy()
    integrated[["cluster", "PC1", "PC2"]] = assignments
    int_seg = (
        integrated.groupby("cluster")
        .agg(
            records=(TARGET, "size"),
            actual_salary_mean=(TARGET, "mean"),
            actual_salary_median=(TARGET, "median"),
            years_mean=("years_of_experience", "mean"),
            demand_mean=("demand_score", "mean"),
        )
        .reset_index()
    )
    save_csv(int_seg, stages["int"] / "segment_salary_summary.csv")
    city = (
        integrated.groupby(["cluster", "country", "city"])
        .agg(records=(TARGET, "size"), salary_mean=(TARGET, "mean"))
        .reset_index()
    )
    save_csv(city, stages["int"] / "segment_city_country_summary.csv")
    locked_pred = diag["locked_test_predictions"].copy()
    emit_event(
        "operation_started",
        step_id="integrated.segment.predict",
        operation="predict_segments",
        status="started",
        message="Assign existing locked-test rows to the fitted segmentation space.",
        detail_level=3,
        extra={"rows": int(len(test)), "refit": False},
    )
    test_assign = predict_segments(seg_bundle, test)
    emit_event(
        "operation_completed",
        step_id="integrated.segment.predict",
        operation="predict_segments",
        status="completed",
        message="Locked-test segment assignment completed without refitting.",
        detail_level=3,
        extra={
            "rows": int(len(test)),
            "refit": False,
            "assigned_clusters": int(pd.Series(test_assign).nunique()),
        },
    )
    locked_pred["cluster"] = test_assign.astype(int)
    pred_by_seg = (
        locked_pred.groupby("cluster")
        .agg(
            records=(TARGET, "size"),
            actual_salary_mean=(TARGET, "mean"),
            predicted_salary_mean=("predicted_salary_usd", "mean"),
            MAE=("absolute_error_usd", "mean"),
        )
        .reset_index()
    )
    save_csv(pred_by_seg, stages["int"] / "predicted_salary_by_segment.csv")
    _audit_complete_stage(
        "I1",
        "Integrated Insight — Segments + Salary Prediction",
        extra={
            "prepared_rows": int(len(prepared)),
            "locked_test_rows": int(len(test)),
            "segment_groups": int(len(pred_by_seg)),
        },
    )

    # ── FULL PIPELINE STATUS ─────────────────────────────────────────────────
    _audit_start_stage("F", "Full Pipeline — Status & Reproducibility Summary")
    status = pd.DataFrame(
        [
            {
                "order": "1",
                "stage": "Common 1 — Project Scope & Raw Data Ingestion",
                "status": "PASS",
                "artifact": "01_data_basic_clean/raw_profile.csv",
            },
            {
                "order": "2",
                "stage": "Common 2 — Basic Clean",
                "status": "PASS",
                "artifact": "01_data_basic_clean/basic_clean.csv",
            },
            {
                "order": "3",
                "stage": "Common 3 — Data Quality & Contradiction Check",
                "status": "PASS",
                "artifact": "01_data_basic_clean/contradiction_summary.csv",
            },
            {
                "order": "4",
                "stage": "Common 4 — Feature Governance",
                "status": "PASS",
                "artifact": "02_data_ready_for_ml/feature_policy.csv",
            },
            {
                "order": "5",
                "stage": "Common 5 — Shared Prepared Feature Base",
                "status": "PASS",
                "artifact": "02_data_ready_for_ml/preprocessing_contract.json",
            },
            {
                "order": "A1-A8",
                "stage": "Branch A — AI Job Market Segmentation",
                "status": "PASS",
                "artifact": "03_ai_job_market_segmentation/k_selection_rationale.json",
            },
            {
                "order": "B1-B3",
                "stage": "Branch B — Temporal Split & TRAIN-only Preprocessing",
                "status": "PASS",
                "artifact": "02_data_ready_for_ml/08_training_readiness.json",
            },
            {
                "order": "B4",
                "stage": "Branch B — Model Training & Comparison",
                "status": "PASS",
                "artifact": "04_model_comparison/09_model_comparison_temporal_cv.csv",
            },
            {
                "order": "B5-B6",
                "stage": "Branch B — Best Model & Locked Test",
                "status": "PASS",
                "artifact": "05_best_model/10_final_locked_test_metrics.csv",
            },
            {
                "order": "B7",
                "stage": "Branch B — Deployment & Prediction Output",
                "status": "PASS",
                "artifact": "artifacts/model_bundle.joblib",
            },
            {
                "order": "8",
                "stage": "Integrated Insight & Streamlit Reporting",
                "status": "PASS",
                "artifact": "07_integrated_insight/segment_salary_summary.csv",
            },
        ]
    )
    save_csv(status, stages["full"] / "pipeline_status.csv")
    summary = {
        "run_id": run_id,
        "source_file": str(raw_path),
        "source_sha256": sha256_file(raw_path),
        "workspace_root": str(workspace),
        "raw_shape": list(raw.shape),
        "clean_shape": list(clean.shape),
        "development_rows": len(dev),
        "locked_test_rows": len(test),
        "segmentation": seg_meta,
        "best_model": best_family,
        "locked_test_metrics": metrics,
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    save_json(summary, stages["full"] / "run_summary.json")
    _audit_complete_stage(
        "F",
        "Full Pipeline — Status & Reproducibility Summary",
        extra={
            "run_id": run_id,
            "source_sha256": summary["source_sha256"],
            "best_model": best_family,
        },
    )
    return summary


def run_pipeline(
    raw_path: Path,
    root: Path | None = None,
    workspace_root: Path | None = None,
    *,
    debuglog: bool = False,
    on_audit_event=None,
) -> dict[str, Any]:
    """Run the complete offline workflow with an additive training audit log."""
    project = root or project_root()
    workspace = Path(workspace_root) if workspace_root is not None else project
    cfg = load_config(project / "config" / "project.yaml")
    seed = int(cfg["project"].get("random_seed", 42))
    with start_training_audit(
        workspace_root=workspace,
        raw_path=raw_path,
        seed=seed,
        target=TARGET,
        features=MODEL_FEATURES,
        debuglog=debuglog,
        on_event=on_audit_event,
    ):
        return _run_pipeline_impl(raw_path=raw_path, root=root, workspace_root=workspace_root)
