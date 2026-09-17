from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SEG = ROOT / "outputs" / "03_ai_job_market_segmentation"


def test_five_representations_are_persisted():
    df = pd.read_csv(SEG / "representation_summary.csv")
    assert set(df["representation_id"]) == {
        "R0_GLOBAL_PCA",
        "R1_FAMILYWISE_PCA",
        "R2_NO_JOB_CATEGORY",
        "R3_NO_YEARS_EXPERIENCE",
        "R4_NO_JOB_CATEGORY_NO_YEARS",
    }
    assert df["selected_representation"].sum() == 1


def test_selected_input_contract_matches_ablation_decision():
    meta = json.loads((SEG / "segmentation_metadata.json").read_text(encoding="utf-8"))
    rep_id = meta["representation_id"]
    selected = meta["representation_input_features"]
    excluded = meta["representation_excluded_features"]
    if rep_id == "R2_NO_JOB_CATEGORY":
        assert "job_category" not in selected
        assert "job_category" in excluded
    if rep_id == "R3_NO_YEARS_EXPERIENCE":
        assert "years_of_experience" not in selected
        assert "years_of_experience" in excluded
    if rep_id == "R4_NO_JOB_CATEGORY_NO_YEARS":
        assert "job_category" not in selected
        assert "years_of_experience" not in selected
        assert "job_category" in excluded
        assert "years_of_experience" in excluded


def test_representation_pairwise_ari_is_complete():
    ari = pd.read_csv(SEG / "representation_pairwise_ari.csv")
    assert len(ari) == 25
    assert ari["ari"].between(-1, 1).all()


def test_component_loadings_are_offline_evidence():
    loads = pd.read_csv(SEG / "representation_component_loadings.csv")
    assert {"representation_id", "component", "encoded_feature", "absolute_loading"}.issubset(
        loads.columns
    )
    assert (loads["absolute_loading"] >= 0).all()


def test_streamlit_branch_a_does_not_fit_models():
    text = (ROOT / "src" / "pages" / "page03_segmentation.py").read_text(encoding="utf-8")
    forbidden = [".fit(", "KMeans(", "GaussianMixture(", "PCA(", "normalize_skills("]
    for token in forbidden:
        assert token not in text


def test_pca_sensitivity_is_persisted_as_diagnostic():
    sens = pd.read_csv(SEG / "representation_pca_sensitivity.csv")
    assert {0.80, 0.85, 0.90}.issubset(set(round(float(x), 2) for x in sens["variance_threshold"]))
    assert {"representation_id", "family", "retained_components", "variance_retained"}.issubset(
        sens.columns
    )


def test_joint_ablation_dependency_is_persisted():
    dep = pd.read_csv(SEG / "feature_dependency_summary.csv")
    assert (dep["ablation_representation"] == "R4_NO_JOB_CATEGORY_NO_YEARS").any()


def test_branch_a_canonical_input_contract_is_exactly_stage3_primary_keep_13():
    expected = [
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
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from ai_job_market.core import SEGMENTATION_FEATURES

    assert SEGMENTATION_FEATURES == expected
    assert len(SEGMENTATION_FEATURES) == 13
    blocked = {"experience_level", "ai_salary_premium_pct", "demand_growth_yoy_pct"}
    assert blocked.isdisjoint(SEGMENTATION_FEATURES)


def test_r0_r4_ablate_only_from_canonical_13_contract():
    df = pd.read_csv(SEG / "representation_summary.csv").set_index("representation_id")
    assert int(df.loc["R0_GLOBAL_PCA", "raw_input_count"]) == 13
    assert int(df.loc["R1_FAMILYWISE_PCA", "raw_input_count"]) == 13
    assert int(df.loc["R2_NO_JOB_CATEGORY", "raw_input_count"]) == 12
    assert int(df.loc["R3_NO_YEARS_EXPERIENCE", "raw_input_count"]) == 12
    assert int(df.loc["R4_NO_JOB_CATEGORY_NO_YEARS", "raw_input_count"]) == 11
