from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd


NormalizeSkills = Callable[[Any], list[str]]


def _profile_categorical(assignments: pd.DataFrame, column: str) -> pd.DataFrame:
    g = (
        assignments.groupby(["cluster", column], dropna=False)
        .agg(
            records=("record_id", "size"),
            salary_mean_posthoc=("annual_salary_usd", "mean"),
            demand_mean=("demand_score", "mean"),
            years_mean=("years_of_experience", "mean"),
        )
        .reset_index()
    )
    totals = assignments.groupby("cluster").size().rename("cluster_records")
    g = g.merge(totals, on="cluster", how="left", validate="many_to_one")
    g["share_within_cluster"] = g["records"] / g["cluster_records"]
    return g


def _skill_profile(assignments: pd.DataFrame, normalize_skills: NormalizeSkills) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for rec in assignments[["record_id", "cluster", "required_skills", "annual_salary_usd"]].itertuples(index=False):
        # Count a skill once per posting. This exactly matches the intended
        # multi-hot semantics even if an input string repeats a token.
        for skill in set(normalize_skills(rec.required_skills)):
            rows.append(
                {
                    "record_id": int(rec.record_id),
                    "cluster": int(rec.cluster),
                    "skill": skill,
                    "annual_salary_usd": float(rec.annual_salary_usd),
                }
            )
    if not rows:
        return pd.DataFrame(columns=["cluster", "skill", "records", "share_within_cluster", "salary_mean_posthoc"])
    long = pd.DataFrame(rows)
    g = (
        long.groupby(["cluster", "skill"])
        .agg(records=("record_id", "nunique"), salary_mean_posthoc=("annual_salary_usd", "mean"))
        .reset_index()
    )
    totals = assignments.groupby("cluster").size().rename("cluster_records")
    g = g.merge(totals, on="cluster", how="left", validate="many_to_one")
    g["share_within_cluster"] = g["records"] / g["cluster_records"]
    return g.sort_values(["cluster", "records", "skill"], ascending=[True, False, True]).reset_index(drop=True)


def build_segmentation_evidence(
    assignments: pd.DataFrame,
    candidate_assignments: pd.DataFrame,
    family_balance_diagnostics: pd.DataFrame,
    normalize_skills: NormalizeSkills,
) -> dict[str, pd.DataFrame]:
    """Create every analytical table required by the Streamlit segmentation page.

    Streamlit may filter rows for presentation, but tokenization, aggregation,
    profiling and candidate-label joins are performed here in the offline
    pipeline so the UI remains an evidence renderer rather than an analytical
    compute layer.
    """

    required = {
        "record_id", "cluster", "PC1", "PC2", "job_title", "job_category",
        "country", "city", "experience_level", "years_of_experience",
        "company_size", "industry", "remote_work", "demand_score",
        "demand_growth_yoy_pct", "benefits_score_10", "ai_salary_premium_pct",
        "required_skills", "annual_salary_usd",
    }
    missing = sorted(required.difference(assignments.columns))
    if missing:
        raise ValueError(f"Cannot build segmentation evidence; assignments missing: {missing}")

    out: dict[str, pd.DataFrame] = {}
    out["family_balance_diagnostics"] = family_balance_diagnostics.copy()
    out["cluster_job_category_profile"] = _profile_categorical(assignments, "job_category")
    out["cluster_experience_level_profile"] = _profile_categorical(assignments, "experience_level")

    out["cluster_years_distribution"] = (
        assignments.groupby(["cluster", "years_of_experience"])
        .size().rename("records").reset_index()
    )

    company_parts = []
    for col in ["company_size", "industry", "remote_work"]:
        x = _profile_categorical(assignments, col).rename(columns={col: "category"})
        x.insert(1, "feature", col)
        company_parts.append(x)
    out["cluster_company_profile"] = pd.concat(company_parts, ignore_index=True)

    geo_parts = []
    for col in ["country", "city"]:
        x = _profile_categorical(assignments, col).rename(columns={col: "category"})
        x.insert(1, "geo_level", col)
        geo_parts.append(x)
    out["cluster_geography_profile"] = pd.concat(geo_parts, ignore_index=True)

    out["cluster_market_signal_values"] = assignments[
        [
            "record_id", "cluster", "job_category", "country",
            "demand_score", "demand_growth_yoy_pct", "benefits_score_10",
            "ai_salary_premium_pct",
        ]
    ].copy()

    out["cluster_skill_profile"] = _skill_profile(assignments, normalize_skills)

    out["country_cluster_profile"] = (
        assignments.groupby(["cluster", "country"])
        .agg(records=("record_id", "size"), salary_mean_posthoc=("annual_salary_usd", "mean"))
        .reset_index()
    )
    out["city_cluster_profile"] = (
        assignments.groupby(["cluster", "country", "city"])
        .agg(records=("record_id", "size"), salary_mean_posthoc=("annual_salary_usd", "mean"))
        .reset_index()
    )

    hover_cols = [
        "record_id", "PC1", "PC2", "job_title", "job_category", "country", "city",
        "years_of_experience", "demand_score",
    ]
    base = assignments[hover_cols].copy()
    coords = candidate_assignments.merge(base, on="record_id", how="left", validate="many_to_one")
    out["candidate_cluster_coordinates"] = coords
    bal = (
        candidate_assignments.groupby(["algorithm", "k", "cluster"])
        .size().rename("records").reset_index()
    )
    totals = bal.groupby(["algorithm", "k"])["records"].transform("sum")
    bal["share"] = bal["records"] / totals
    out["candidate_cluster_balance"] = bal

    return out


def build_segmentation_insights(
    *,
    meta: dict[str, Any],
    rationale: dict[str, Any],
    profiles: pd.DataFrame,
    evidence: dict[str, pd.DataFrame],
) -> dict[str, dict[str, str]]:
    """Generate run-specific narrative evidence once in the offline pipeline."""

    family = evidence["family_balance_diagnostics"].copy()
    biggest = family.sort_values("encoded_dimensions", ascending=False).iloc[0]
    pca_cluster_pct = 100.0 * float(meta["clustering_variance_captured"])
    pca_visual_pct = 100.0 * float(meta["visualization_variance_captured"])

    largest = profiles.sort_values("records", ascending=False).iloc[0]
    salary_spread = float(profiles["salary_mean"].max() - profiles["salary_mean"].min()) if len(profiles) > 1 else 0.0

    job = evidence["cluster_job_category_profile"]
    job_top = job.sort_values(["cluster", "records"], ascending=[True, False]).groupby("cluster", as_index=False).head(1)
    job_text = "; ".join(f"C{int(r.cluster)}: {r.job_category} ({int(r.records)} records)" for r in job_top.itertuples())

    skill = evidence["cluster_skill_profile"]
    if len(skill):
        skill_top = skill.sort_values(["cluster", "records"], ascending=[True, False]).groupby("cluster", as_index=False).head(1)
        skill_text = "; ".join(f"C{int(r.cluster)}: {r.skill} ({int(r.records)})" for r in skill_top.itertuples())
    else:
        skill_text = "No normalized skill tokens were available."

    country = evidence["country_cluster_profile"].groupby("country", as_index=False)["records"].sum().sort_values("records", ascending=False)
    if len(country):
        top_country = country.iloc[0]
        geo_obs = f"{top_country.country} has the largest run-level posting volume ({int(top_country.records)} records)."
    else:
        geo_obs = "No country-level evidence is available."

    prof = profiles.copy()
    demand_spread = float(prof["demand_mean"].max() - prof["demand_mean"].min()) if len(prof) > 1 else 0.0
    years_spread = float(prof["years_mean"].max() - prof["years_mean"].min()) if len(prof) > 1 else 0.0

    return {
        "representation": {
            "observed": (
                f"{biggest.family} expands to {int(biggest.encoded_dimensions)} encoded dimensions. "
                f"Clustering uses PC1..PC{int(meta['pca_components_for_clustering'])}, capturing {pca_cluster_pct:.1f}% of DEV variance; "
                f"PC1+PC2 are visualization-only and capture {pca_visual_pct:.1f}%."
            ),
            "interpretation": "The clustering space is explicitly separated from the 2D display space, so visual overlap in PC1/PC2 is not treated as the full clustering geometry.",
            "action": "Recompute the PCA component count on every new dataset using the configured cumulative-variance threshold; never hard-code two PCs for clustering.",
            "tone": "info",
        },
        "k_selection": {
            "observed": rationale["decision_note"],
            "interpretation": (
                f"The official candidate passes seed stability, subsample stability and minimum-cluster-size gates before silhouette/parsimony selection. "
                f"Mean subsample ARI is {float(rationale['selected_resample_stability_ari_mean']):.3f}."
            ),
            "action": "Treat the selected K as a descriptive structure for this run and re-run all stability gates on new data or later periods.",
            "tone": "success" if float(rationale["selected_silhouette"]) >= 0.35 else "warning",
        },
        "profiles": {
            "observed": f"Largest official cluster is C{int(largest.cluster)} with {int(largest.records)} records ({float(largest.share_pct):.1f}%); post-hoc mean-salary spread is ${salary_spread:,.0f}.",
            "interpretation": "Salary is descriptive after clustering and did not contribute to cluster formation.",
            "action": "Name segments only after checking the structural family profiles and stability evidence.",
            "tone": "warning",
        },
        "Job Domain": {
            "observed": f"Run-level dominant job categories by cluster — {job_text}.",
            "interpretation": "Job-domain concentration indicates which role taxonomy contributes most to each structural segment.",
            "action": "Use the category profile together with skills and demand before assigning business labels.",
            "tone": "info",
        },
        "Skills": {
            "observed": f"Most frequent normalized skill token by cluster — {skill_text}.",
            "interpretation": "Skill frequency is based on deduplicated posting-level multi-hot semantics generated offline.",
            "action": "Check whether the same skill signature persists under later data and resampling before operationalizing a segment label.",
            "tone": "info",
        },
        "Experience": {
            "observed": f"Mean years-of-experience differs by up to {years_spread:.1f} years across official clusters.",
            "interpretation": "Experience contributes to structural separation, but this dataset has known synthetic-looking experience semantics.",
            "action": "Interpret experience descriptively and avoid turning the observed pattern into a labor-market rule.",
            "tone": "warning",
        },
        "Company": {
            "observed": "Company-size, industry and work-mode profiles are pre-aggregated by cluster in the offline evidence layer.",
            "interpretation": "These distributions show organizational composition of the segments, not causal drivers.",
            "action": "Use whichever company field shows a stable and material concentration difference across clusters.",
            "tone": "info",
        },
        "Geography": {
            "observed": geo_obs,
            "interpretation": "Geographic concentration can influence cluster composition even though salary is excluded from clustering.",
            "action": "Inspect country and city concentration before generalizing a segment across markets.",
            "tone": "info",
        },
        "Demand / Benefits": {
            "observed": f"Mean demand score differs by {demand_spread:.1f} points across official clusters.",
            "interpretation": "Market-signal differences help explain structural segmentation but remain descriptive associations.",
            "action": "Re-check demand/benefit distributions on refreshed data because market signals can drift quickly.",
            "tone": "info",
        },
        "geography": {
            "observed": geo_obs,
            "interpretation": "Country and city views describe where the current segment assignments are concentrated.",
            "action": "Use geography as a context layer, not as proof that a cluster is universally valid across countries.",
            "tone": "info",
        },
    }
