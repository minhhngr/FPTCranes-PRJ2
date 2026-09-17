from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SEG = ROOT / "outputs" / "03_ai_job_market_segmentation"


def test_subsample_stability_is_persisted_and_used_as_gate():
    ev = pd.read_csv(SEG / "cluster_evaluation.csv")
    rationale = json.load(open(SEG / "k_selection_rationale.json", encoding="utf-8"))
    runs = pd.read_csv(SEG / "resample_stability_runs.csv")
    required = {
        "resample_stability_ari_mean",
        "resample_stability_ari_std",
        "resample_stability_ari_p10",
        "resample_stability_gate",
        "eligible",
        "selected",
    }
    assert required.issubset(ev.columns)
    assert len(runs) > 0
    assert {"algorithm", "k", "resample", "ari_vs_full_reference"}.issubset(runs.columns)
    selected = ev.loc[ev["selected"].astype(bool)].iloc[0]
    assert float(selected["resample_stability_ari_mean"]) >= float(rationale["resample_stability_threshold"])


def test_pca_cluster_space_is_distinct_from_2d_visualization():
    meta = json.load(open(SEG / "segmentation_metadata.json", encoding="utf-8"))
    pca = pd.read_csv(SEG / "pca_variance.csv")
    assert meta["pca_components_for_clustering"] >= 2
    assert meta["clustering_variance_captured"] >= meta["pca_variance_threshold"] - 1e-12
    assert meta["visualization_components"] == 2
    assert pca["used_for_clustering"].sum() == meta["pca_components_for_clustering"]
    assert pca["used_for_visualization"].sum() == 2
    assert "visualization" in meta["visualization_space"].lower()


def test_streamlit_segmentation_page_is_evidence_only():
    page = (ROOT / "src/pages/page03_segmentation.py").read_text(encoding="utf-8")
    assert "normalize_skills" not in page
    assert ".groupby(" not in page
    assert ".fit(" not in page
    assert "cluster_skill_profile.csv" not in page  # loaded through evidence name helper
    assert "segmentation_insights.json" in page


def test_offline_segmentation_evidence_contract_exists():
    expected = [
        "family_balance_diagnostics.csv",
        "pca_variance.csv",
        "resample_stability_runs.csv",
        "candidate_cluster_coordinates.csv",
        "candidate_cluster_balance.csv",
        "cluster_job_category_profile.csv",
        "cluster_skill_profile.csv",
        "cluster_experience_level_profile.csv",
        "cluster_years_distribution.csv",
        "cluster_company_profile.csv",
        "cluster_geography_profile.csv",
        "cluster_market_signal_values.csv",
        "country_cluster_profile.csv",
        "city_cluster_profile.csv",
        "segmentation_insights.json",
    ]
    for name in expected:
        assert (SEG / name).exists(), name
