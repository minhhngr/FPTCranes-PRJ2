from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in [str(ROOT), str(ROOT / "src")]:
    while _p in sys.path:
        sys.path.remove(_p)
sys.path.insert(0, str(ROOT / "src"))
sys.path.append(str(ROOT))
if "pipeline" in sys.modules and not hasattr(sys.modules["pipeline"], "__path__"):
    del sys.modules["pipeline"]
from ai_job_market.core import MODEL_FEATURES, SOURCE_COLUMNS, TARGET, normalize_skills


def test_source_schema_and_basic_clean_shape():
    raw = pd.read_csv(ROOT / "data/raw/ai_jobs_market_2025_2026.csv")
    clean = pd.read_csv(ROOT / "outputs/01_data_basic_clean/basic_clean.csv")
    assert set(SOURCE_COLUMNS).issubset(raw.columns)
    assert clean.shape == (1499, 24)
    assert "job_id" not in clean and "skill_count" not in clean


def test_temporal_holdout_isolated():
    dev = pd.read_csv(ROOT / "outputs/02_data_ready_for_ml/development_raw.csv")
    test = pd.read_csv(ROOT / "outputs/02_data_ready_for_ml/locked_test_raw.csv")
    s = json.load(
        open(ROOT / "outputs/02_data_ready_for_ml/temporal_split_summary.json", encoding="utf-8")
    )
    y, m = s["locked_test_year"], s["locked_test_month"]
    assert ((test.posting_year == y) & (test.posting_month == m)).all()
    assert not ((dev.posting_year == y) & (dev.posting_month == m)).any()


def test_model_contract_has_no_target_leakage():
    meta = json.load(open(ROOT / "artifacts/metadata.json", encoding="utf-8"))
    assert meta["model_features"] == MODEL_FEATURES
    assert TARGET not in meta["model_features"]
    for c in ["salary_min_usd", "salary_max_usd", "salary_tier"]:
        assert c not in meta["model_features"]


def test_skill_vocabulary_and_count_logic():
    meta = json.load(open(ROOT / "artifacts/metadata.json", encoding="utf-8"))
    assert len(meta["skill_vocabulary"]) > 0
    assert normalize_skills("Python|SQL|Python") == ["Python", "SQL"]


def test_saved_bundle_predicts_raw_rows():
    meta = json.load(open(ROOT / "artifacts/metadata.json", encoding="utf-8"))
    test = pd.read_csv(ROOT / "outputs/02_data_ready_for_ml/locked_test_raw.csv")
    bundle = joblib.load(ROOT / "artifacts/model_bundle.joblib")
    p = np.asarray(bundle.predict(test[meta["model_features"]].head(3)), dtype=float)
    assert p.shape == (3,)
    assert np.isfinite(p).all()


def test_segmentation_target_excluded_and_dimension_traceable():
    meta = json.load(
        open(
            ROOT / "outputs/03_ai_job_market_segmentation/segmentation_metadata.json",
            encoding="utf-8",
        )
    )
    dims = pd.read_csv(ROOT / "outputs/03_ai_job_market_segmentation/family_dimensions.csv")
    assert meta["target_used_for_clustering"] is False
    assert meta["encoded_dimensions"] == int(dims.encoded_dimensions.sum())


def test_validation_report_passes():
    report = json.load(open(ROOT / "outputs/validation_report.json", encoding="utf-8"))
    assert report["overall"] == "PASS"
    assert report["failed"] == 0


def test_model_comparison_includes_required_baselines_and_runtime():
    comp = pd.read_csv(ROOT / "outputs/04_model_comparison/model_comparison.csv")
    expected = {
        "Dummy Median",
        "Linear Regression",
        "Ridge Regression",
        "Random Forest",
        "Gradient Boosting",
    }
    assert expected.issubset(set(comp.model))
    assert {"fit_time_mean_s", "predict_time_mean_s"}.issubset(comp.columns)
    assert (comp[["fit_time_mean_s", "predict_time_mean_s"]] >= 0).all().all()


def test_interactive_evidence_outputs_exist():
    expected = [
        "outputs/01_data_basic_clean/integrity_row_flags.csv",
        "outputs/01_data_basic_clean/numeric_descriptive_summary.csv",
        "outputs/02_data_ready_for_ml/preprocessing_feature_map.csv",
        "outputs/02_data_ready_for_ml/numeric_scaling_summary.csv",
        "outputs/02_data_ready_for_ml/skill_token_summary.csv",
        "outputs/02_data_ready_for_ml/skill_target_correlations.csv",
        "outputs/05_best_model/error_by_job_category.csv",
        "outputs/05_best_model/error_by_country.csv",
    ]
    for rel in expected:
        assert (ROOT / rel).exists(), rel


def test_streamlit_pages_are_output_driven_and_interactive():
    pages = list((ROOT / "src/pages").glob("page*.py"))
    text = "\n".join(p.read_text(encoding="utf-8") for p in pages)
    assert ".fit(" not in text
    assert "st.plotly_chart" in (ROOT / "src/pages/common.py").read_text(encoding="utf-8")
    assert text.count("selectbox(") >= 12
    assert text.count("multiselect(") >= 10


def test_schema_preflight_accepts_source_and_blocks_missing_target():
    from ai_job_market.core import validate_input_schema

    raw = pd.read_csv(ROOT / "data/raw/ai_jobs_market_2025_2026.csv")
    good = validate_input_schema(raw)
    assert good["valid"] is True
    bad = validate_input_schema(raw.drop(columns=[TARGET]))
    assert bad["valid"] is False
    assert TARGET in bad["missing_columns"]


def test_segmentation_k_selection_is_explicit_stable_and_reproducible():
    p = ROOT / "outputs/03_ai_job_market_segmentation"
    ev = pd.read_csv(p / "cluster_evaluation.csv")
    cand = pd.read_csv(p / "candidate_cluster_assignments.csv")
    rationale = json.load(open(p / "k_selection_rationale.json", encoding="utf-8"))
    required = {
        "stability_ari",
        "min_cluster_share",
        "eligible",
        "within_primary_tolerance",
        "selected",
    }
    assert required.issubset(ev.columns)
    expected = {(a, k) for a in ["KMeans", "GMM"] for k in range(2, 9)}
    assert expected.issubset(set(zip(ev.algorithm, ev.k.astype(int))))
    assert ev.selected.sum() == 1
    selected = ev.loc[ev.selected.astype(bool)].iloc[0]
    assert rationale["selected_algorithm"] == selected.algorithm
    assert int(rationale["selected_k"]) == int(selected.k)
    # Candidate labels are persisted so Streamlit can explore alternative K without fitting.
    assert {"algorithm", "k", "record_id", "cluster"}.issubset(cand.columns)
    assert expected.issubset(set(zip(cand.algorithm, cand.k.astype(int))))


def test_branch_b_main_style_evidence_contract_exists():
    expected = [
        "outputs/02_data_ready_for_ml/08_before_after_processing.csv",
        "outputs/02_data_ready_for_ml/08_monthly_distribution.csv",
        "outputs/02_data_ready_for_ml/08_split_summary.csv",
        "outputs/02_data_ready_for_ml/08_training_readiness.json",
        "outputs/04_model_comparison/09_feature_family_ablation.csv",
        "outputs/04_model_comparison/09_feature_importance_by_fold.csv",
        "outputs/04_model_comparison/09_feature_importance_drift.csv",
        "outputs/04_model_comparison/09_model_comparison_fold_metrics.csv",
        "outputs/04_model_comparison/09_model_comparison_temporal_cv.csv",
        "outputs/04_model_comparison/09_model_runtime_performance.csv",
        "outputs/05_best_model/10_best_model_tuning_results.csv",
        "outputs/05_best_model/10_error_slices.csv",
        "outputs/05_best_model/10_final_locked_test_metrics.csv",
        "outputs/05_best_model/10_locked_test_predictions_with_error.csv",
        "outputs/05_best_model/10_raw_feature_permutation_importance.csv",
        "outputs/05_best_model/10_encoded_feature_importance.csv",
        "outputs/05_best_model/11_deployment_artifact_manifest.csv",
        "outputs/06_salary_prediction/12_locked_test_prediction_examples.csv",
        "outputs/06_salary_prediction/12_prediction_summary.csv",
        "artifacts/11_bundle_equivalence.json",
        "artifacts/feature_columns.json",
        "artifacts/preprocessor_ml_ready.joblib",
    ]
    for rel in expected:
        assert (ROOT / rel).exists(), rel


def test_branch_b_compatibility_facades_import():
    from pipeline.best_model_selection_feature_importance_review import finalize_salary_model as fsm
    from pipeline.model_training_comparison import run_feature_family_ablation as rffa
    from pipeline.train_test_split import temporal_split as ts
    from training.model_catalog import candidate_models as cm
    from training.temporal_cv import temporal_cv_splits as tcv

    assert callable(cm) and callable(tcv) and callable(ts) and callable(rffa) and callable(fsm)


def test_streamlit_has_global_upload_full_process_and_commentary_every_page():
    data_source = (ROOT / "src/components/data_source.py").read_text(encoding="utf-8")
    assert "file_uploader(" in data_source
    assert "validate_input_schema" in data_source
    assert "Process full pipeline on this dataset" in data_source
    assert "run_pipeline(" in data_source
    assert 'disabled=not report["valid"]' in data_source
    for p in sorted((ROOT / "src/pages").glob("page*.py")):
        txt = p.read_text(encoding="utf-8")
        assert "interpretation_card(" in txt, p.name


def test_legacy_main_archive_manifest_packaged():
    assert (ROOT / "docs/legacy_reference/FPTCranes-PRJ2-main.rar").exists()
    manifest = pd.read_csv(ROOT / "docs/FPTCranes-PRJ2-main_manifest.csv")
    assert len(manifest) > 100
    names = "\n".join(manifest["name"].astype(str))
    for token in [
        "training/temporal_cv.py",
        "training/ablation_study.py",
        "model_training_comparison.py",
        "best_model_selection_feature_importance_review.py",
    ]:
        assert token in names
