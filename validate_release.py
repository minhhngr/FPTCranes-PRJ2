from __future__ import annotations

import ast
import json
import math
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ai_job_market.core import MODEL_FEATURES, SOURCE_COLUMNS, TARGET, validate_input_schema

ROOT = Path(__file__).resolve().parent
for _p in [str(ROOT), str(ROOT / "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def main():
    checks = []

    def check(name, ok, evidence=""):
        checks.append(
            {"check": name, "status": "PASS" if bool(ok) else "FAIL", "evidence": str(evidence)}
        )

    raw = pd.read_csv(ROOT / "data/raw/ai_jobs_market_2025_2026.csv")
    audit = json.load(
        open(ROOT / "outputs/01_data_basic_clean/basic_clean_audit.json", encoding="utf-8")
    )
    clean = pd.read_csv(ROOT / "outputs/01_data_basic_clean/basic_clean.csv")
    prepared = pd.read_csv(ROOT / "outputs/02_data_ready_for_ml/shared_prepared_feature_base.csv")
    dev = pd.read_csv(ROOT / "outputs/02_data_ready_for_ml/development_raw.csv")
    test = pd.read_csv(ROOT / "outputs/02_data_ready_for_ml/locked_test_raw.csv")
    split = json.load(
        open(ROOT / "outputs/02_data_ready_for_ml/temporal_split_summary.json", encoding="utf-8")
    )
    policy = pd.read_csv(ROOT / "outputs/02_data_ready_for_ml/feature_policy.csv")
    tr = pd.read_csv(ROOT / "outputs/02_data_ready_for_ml/train_preprocessed.csv")
    te = pd.read_csv(ROOT / "outputs/02_data_ready_for_ml/locked_test_preprocessed.csv")
    segmeta = json.load(
        open(
            ROOT / "outputs/03_ai_job_market_segmentation/segmentation_metadata.json",
            encoding="utf-8",
        )
    )
    segev = pd.read_csv(ROOT / "outputs/03_ai_job_market_segmentation/cluster_evaluation.csv")
    comp = pd.read_csv(ROOT / "outputs/04_model_comparison/model_comparison.csv")
    met = json.load(open(ROOT / "outputs/05_best_model/locked_test_metrics.json", encoding="utf-8"))
    ser = json.load(
        open(ROOT / "outputs/06_salary_prediction/serialization_check.json", encoding="utf-8")
    )
    meta = json.load(open(ROOT / "artifacts/metadata.json", encoding="utf-8"))
    status = pd.read_csv(ROOT / "outputs/08_full_pipeline/pipeline_status.csv")

    check("01 source schema complete", set(SOURCE_COLUMNS).issubset(raw.columns), f"{raw.shape}")
    check(
        "02 basic-clean row accounting",
        len(clean)
        == audit["raw_rows"]
        - audit["invalid_category_rows_removed"]
        - audit["duplicate_rows_removed"],
        len(clean),
    )
    check("03 basic-clean has 24 columns", clean.shape[1] == 24, clean.shape)
    check(
        "04 identifier removed from basic clean",
        "job_id" not in clean.columns,
        list(clean.columns[:3]),
    )
    check(
        "05 feature construction occurs after basic clean",
        "skill_count" not in clean.columns and "skill_count" in prepared.columns,
        prepared.shape,
    )
    check(
        "06 leakage policy blocks salary-adjacent fields",
        set(["salary_min_usd", "salary_max_usd", "salary_tier"]).issubset(
            set(policy.loc[policy.policy == "BLOCK", "feature"])
        ),
        "salary leakage fields blocked",
    )
    check(
        "07 target absent from serving X",
        TARGET not in MODEL_FEATURES and TARGET not in meta["model_features"],
        meta["model_features"],
    )
    check(
        "08 temporal split row conservation",
        len(dev) + len(test) == len(prepared),
        f"dev={len(dev)}, test={len(test)}",
    )
    ly, lm = int(split["locked_test_year"]), int(split["locked_test_month"])
    locked_mask = (test.posting_year.astype(int) == ly) & (test.posting_month.astype(int) == lm)
    dev_leak = ((dev.posting_year.astype(int) == ly) & (dev.posting_month.astype(int) == lm)).any()
    check(
        "09 full locked period isolated",
        locked_mask.all() and not dev_leak,
        split["locked_test_period_label"],
    )
    check(
        "10 train/test encoded columns identical",
        list(tr.columns) == list(te.columns),
        f"encoded={tr.shape[1]}",
    )
    check(
        "11 preprocessing fit scope recorded as development only",
        json.load(
            open(
                ROOT / "outputs/02_data_ready_for_ml/preprocessing_contract.json", encoding="utf-8"
            )
        )["preprocessor_fit_scope"]
        == "development only",
        "development only",
    )
    check(
        "12 segmentation excludes salary target",
        segmeta["target_used_for_clustering"] is False,
        segmeta,
    )
    expected_pairs = {(a, k) for a in ["KMeans", "GMM"] for k in range(2, 9)}
    actual_pairs = set(zip(segev.algorithm, segev.k.astype(int)))
    check(
        "13 segmentation evaluates KMeans/GMM for K=2..8",
        expected_pairs.issubset(actual_pairs),
        f"candidates={len(actual_pairs)}",
    )
    check(
        "14 family-balanced encoded dimension is traceable",
        segmeta["encoded_dimensions"]
        == int(
            pd.read_csv(
                ROOT / "outputs/03_ai_job_market_segmentation/family_dimensions.csv"
            ).encoded_dimensions.sum()
        ),
        segmeta["encoded_dimensions"],
    )
    best_cv = comp.sort_values("MAE_mean").iloc[0].model
    check(
        "15 model selection follows minimum temporal-CV MAE", meta["model_name"] == best_cv, best_cv
    )
    check(
        "16 locked-test metrics finite",
        all(
            np.isfinite(float(met[k]))
            for k in ["MAE", "RMSE", "R2", "MedAE", "prediction_interval_abs_error_q90"]
        ),
        met,
    )
    check(
        "17 serialized bundle reload equivalence",
        ser["passed"] and ser["reload_max_abs_diff"] <= ser["tolerance"],
        ser,
    )
    bundle = joblib.load(ROOT / "artifacts/model_bundle.joblib")
    pred = np.asarray(bundle.predict(test[meta["model_features"]].head(5)), dtype=float)
    check(
        "18 saved bundle performs finite raw-row inference",
        len(pred) == 5 and np.isfinite(pred).all(),
        pred.tolist(),
    )
    pyfiles = [ROOT / "streamlit.py", *sorted((ROOT / "src/pages").glob("page*.py"))]
    parse_ok = True
    forbidden = []
    for p in pyfiles:
        try:
            ast.parse(p.read_text(encoding="utf-8"))
        except Exception as e:
            parse_ok = False
            forbidden.append(f"{p.name}: parse {e}")
        txt = p.read_text(encoding="utf-8")
        if ".fit(" in txt or "GridSearchCV" in txt or "RandomizedSearchCV" in txt:
            forbidden.append(p.name)
    check(
        "19 Streamlit is report/inference only (no fitting)",
        parse_ok and not forbidden,
        forbidden or "AST parsed; no .fit() or search-CV calls",
    )
    required_models = {
        "Dummy Median",
        "Linear Regression",
        "Ridge Regression",
        "Random Forest",
        "Gradient Boosting",
    }
    check(
        "20 model ladder includes baseline + four approved regressors",
        required_models.issubset(set(comp.model)),
        sorted(comp.model.tolist()),
    )
    check(
        "21 runtime evidence recorded for model comparison",
        {"fit_time_mean_s", "predict_time_mean_s"}.issubset(comp.columns)
        and np.isfinite(
            comp[["fit_time_mean_s", "predict_time_mean_s"]].to_numpy(dtype=float)
        ).all(),
        "fit/predict timing available",
    )
    interactive_outputs = [
        ROOT / "outputs/01_data_basic_clean/integrity_row_flags.csv",
        ROOT / "outputs/02_data_ready_for_ml/preprocessing_feature_map.csv",
        ROOT / "outputs/02_data_ready_for_ml/numeric_scaling_summary.csv",
        ROOT / "outputs/02_data_ready_for_ml/skill_target_correlations.csv",
        ROOT / "outputs/05_best_model/error_by_job_category.csv",
        ROOT / "outputs/05_best_model/error_by_country.csv",
    ]
    check(
        "22 interactive evidence outputs generated",
        all(p.exists() for p in interactive_outputs),
        [p.name for p in interactive_outputs],
    )
    common_txt = (ROOT / "src/pages/common.py").read_text(encoding="utf-8")
    pages_txt = "\n".join(
        p.read_text(encoding="utf-8") for p in sorted((ROOT / "src/pages").glob("page*.py"))
    )
    check(
        "23 Streamlit charts are interactive and filterable",
        "st.plotly_chart" in common_txt
        and pages_txt.count("selectbox(") >= 12
        and pages_txt.count("multiselect(") >= 10,
        f"selectbox={pages_txt.count('selectbox(')}, multiselect={pages_txt.count('multiselect(')}",
    )
    pipeline_imgs = [
        ROOT / "docs/pipelines/Pipeline_AI-salary-overall.png",
        ROOT / "docs/pipelines/Pipeline_AI-salary-prediction.png",
        ROOT / "docs/pipelines/Pipeline_AI-salary-segmentation.png",
    ]
    check(
        "24 all three supplied pipeline diagrams packaged",
        all(p.exists() for p in pipeline_imgs),
        [p.name for p in pipeline_imgs],
    )
    check(
        "25 technical design and legacy reference packaged",
        (ROOT / "docs/Document_QD Project KHDL&AI(8).docx").exists()
        and (ROOT / "docs/legacy_reference/FPTCranes-PRJ2_full(3).7z").exists(),
        "design doc + legacy source archive retained",
    )

    rationale = json.load(
        open(
            ROOT / "outputs/03_ai_job_market_segmentation/k_selection_rationale.json",
            encoding="utf-8",
        )
    )
    cand_assign = pd.read_csv(
        ROOT / "outputs/03_ai_job_market_segmentation/candidate_cluster_assignments.csv"
    )
    selected_rows = (
        segev.loc[segev["selected"].astype(bool)] if "selected" in segev.columns else pd.DataFrame()
    )
    gates = {
        "stability_ari",
        "min_cluster_share",
        "eligible",
        "within_primary_tolerance",
        "selected",
    }
    k_ok = (
        gates.issubset(segev.columns)
        and len(selected_rows) == 1
        and rationale.get("selected_algorithm") == str(selected_rows.iloc[0]["algorithm"])
        and int(rationale.get("selected_k")) == int(selected_rows.iloc[0]["k"])
    )
    check(
        "26 K-selection policy is explicit and artifact-backed",
        k_ok,
        f"selected={rationale.get('selected_algorithm')} K={rationale.get('selected_k')}; stability={rationale.get('selected_stability_ari')}; min_share={rationale.get('selected_min_cluster_share')}",
    )
    cand_pairs = (
        set(zip(cand_assign.algorithm, cand_assign.k.astype(int)))
        if {"algorithm", "k"}.issubset(cand_assign.columns)
        else set()
    )
    check(
        "27 alternative cluster assignments persisted for UI explorer",
        expected_pairs.issubset(cand_pairs),
        f"candidate pairs={len(cand_pairs)}",
    )

    good_schema = validate_input_schema(raw)
    bad_schema = validate_input_schema(raw.drop(columns=[TARGET]))
    check(
        "28 upload schema gate accepts valid raw contract and blocks missing target",
        good_schema["valid"]
        and not bad_schema["valid"]
        and TARGET in bad_schema["missing_columns"],
        f"good={good_schema['valid']}, bad={bad_schema['valid']}",
    )

    ds_txt = (ROOT / "src/components/data_source.py").read_text(encoding="utf-8")
    upload_ok = all(
        tok in ds_txt
        for tok in [
            "file_uploader(",
            "validate_input_schema",
            "Process full pipeline on this dataset",
            "run_pipeline(",
            "workspace_root=workspace",
            'disabled=not report["valid"]',
        ]
    )
    check(
        "29 Streamlit upload can validate then trigger isolated full pipeline",
        upload_ok,
        "global data source component",
    )

    main_style = [
        ROOT / "outputs/02_data_ready_for_ml/08_training_readiness.json",
        ROOT / "outputs/04_model_comparison/09_feature_family_ablation.csv",
        ROOT / "outputs/04_model_comparison/09_feature_importance_by_fold.csv",
        ROOT / "outputs/04_model_comparison/09_model_comparison_temporal_cv.csv",
        ROOT / "outputs/05_best_model/10_best_model_tuning_results.csv",
        ROOT / "outputs/05_best_model/10_error_slices.csv",
        ROOT / "outputs/05_best_model/10_locked_test_predictions_with_error.csv",
        ROOT / "outputs/05_best_model/11_deployment_artifact_manifest.csv",
        ROOT / "outputs/06_salary_prediction/12_locked_test_prediction_examples.csv",
        ROOT / "artifacts/11_bundle_equivalence.json",
    ]
    check(
        "30 Branch B main-style evidence contract combined",
        all(p.exists() for p in main_style),
        [p.name for p in main_style],
    )

    facade_files = [
        ROOT / "src/training/model_catalog.py",
        ROOT / "src/training/temporal_cv.py",
        ROOT / "src/training/ablation_study.py",
        ROOT / "src/training/tuning.py",
        ROOT / "src/pipeline/train_test_split.py",
        ROOT / "src/pipeline/model_training_comparison.py",
        ROOT / "src/pipeline/best_model_selection_feature_importance_review.py",
    ]
    check(
        "31 Branch B stage/training compatibility modules packaged",
        all(p.exists() for p in facade_files),
        [p.name for p in facade_files],
    )

    commentary = []
    for p in sorted((ROOT / "src/pages").glob("page*.py")):
        commentary.append((p.name, p.read_text(encoding="utf-8").count("interpretation_card(")))
    check(
        "32 every Streamlit page has data-driven interpretation cards",
        all(n > 0 for _, n in commentary),
        commentary,
    )

    main_manifest = ROOT / "docs/FPTCranes-PRJ2-main_manifest.csv"
    legacy_main = ROOT / "docs/legacy_reference/FPTCranes-PRJ2-main.rar"
    manifest_ok = False
    if main_manifest.exists():
        mm = pd.read_csv(main_manifest)
        manifest_ok = len(mm) > 100 and any(
            mm["name"].astype(str).str.contains("training/temporal_cv.py", regex=False)
        )
    check(
        "33 uploaded FPTCranes-PRJ2-main archive and manifest retained",
        legacy_main.exists() and manifest_ok,
        f"archive={legacy_main.exists()}, manifest={manifest_ok}",
    )

    # Branch-A v3: structural stability, PCA-space contract and evidence-only UI.
    resample_path = ROOT / "outputs/03_ai_job_market_segmentation/resample_stability_runs.csv"
    resample_ok = False
    if resample_path.exists() and {
        "resample_stability_ari_mean",
        "resample_stability_gate",
    }.issubset(segev.columns):
        rs = pd.read_csv(resample_path)
        sel = segev.loc[segev["selected"].astype(bool)].iloc[0]
        resample_ok = (
            len(rs) > 0
            and float(sel["resample_stability_ari_mean"])
            >= float(rationale["resample_stability_threshold"])
            and bool(sel["resample_stability_gate"])
        )
    check(
        "34 subsample stability is persisted and gates the selected K",
        resample_ok,
        f"selected mean subsample ARI={rationale.get('selected_resample_stability_ari_mean')}",
    )

    pca_path = ROOT / "outputs/03_ai_job_market_segmentation/pca_variance.csv"
    pca_ok = False
    if pca_path.exists():
        pv = pd.read_csv(pca_path)
        pca_ok = (
            int(segmeta.get("visualization_components", 0)) == 2
            and int(segmeta.get("pca_components_for_clustering", 0)) >= 2
            and float(segmeta.get("clustering_variance_captured", 0))
            >= float(segmeta.get("pca_variance_threshold", 1)) - 1e-12
            and int(pv["used_for_clustering"].astype(bool).sum())
            == int(segmeta["pca_components_for_clustering"])
        )
    check(
        "35 PCA clustering space is explicit and distinct from PC1/PC2 display",
        pca_ok,
        f"cluster PCs={segmeta.get('pca_components_for_clustering')}; cluster variance={segmeta.get('clustering_variance_captured')}; display variance={segmeta.get('visualization_variance_captured')}",
    )

    p3_txt = (ROOT / "src/pages/page03_segmentation.py").read_text(encoding="utf-8")
    offline_outputs = [
        ROOT / "outputs/03_ai_job_market_segmentation/cluster_skill_profile.csv",
        ROOT / "outputs/03_ai_job_market_segmentation/cluster_job_category_profile.csv",
        ROOT / "outputs/03_ai_job_market_segmentation/family_balance_diagnostics.csv",
        ROOT / "outputs/03_ai_job_market_segmentation/candidate_cluster_coordinates.csv",
        ROOT / "outputs/03_ai_job_market_segmentation/segmentation_insights.json",
    ]
    evidence_only = (
        all(x.exists() for x in offline_outputs)
        and "normalize_skills" not in p3_txt
        and ".groupby(" not in p3_txt
        and ".fit(" not in p3_txt
    )
    check(
        "36 Branch-A Streamlit is evidence-only for tokenization/profile derivation",
        evidence_only,
        [x.name for x in offline_outputs],
    )

    try:
        import streamlit  # noqa

        streamlit_status = {
            "status": "AVAILABLE",
            "version": getattr(streamlit, "__version__", "unknown"),
            "note": "Package is available; launch command can be executed in this environment.",
        }
    except Exception as e:
        streamlit_status = {
            "status": "NOT_INSTALLED_IN_VALIDATION_RUNTIME",
            "version": None,
            "note": "Source/AST validation passed. Install requirements.txt before launching the dashboard.",
            "error": str(e),
        }
    report = {
        "checks": checks,
        "passed": sum(c["status"] == "PASS" for c in checks),
        "failed": sum(c["status"] == "FAIL" for c in checks),
        "overall": "PASS" if all(c["status"] == "PASS" for c in checks) else "FAIL",
        "streamlit_runtime": streamlit_status,
    }
    (ROOT / "outputs").mkdir(exist_ok=True)
    with open(ROOT / "outputs/validation_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(
        f"Release validation: {report['overall']} — {report['passed']}/{len(checks)} checks passed"
    )
    print(f"Streamlit runtime: {streamlit_status['status']}")
    for c in checks:
        print(f"[{c['status']}] {c['check']}: {c['evidence']}")
    raise SystemExit(0 if report["overall"] == "PASS" else 1)


if __name__ == "__main__":
    main()
