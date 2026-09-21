from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from ai_job_market.training_evidence_io import required_evidence_files, sha256_file
from pages.training_validation_presentation import (
    discover_latest_training_validation,
    load_training_validation_view,
    sort_training_events,
)

ROOT = Path(__file__).resolve().parents[1]


def _write_csv(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def build_complete_pack(
    root: Path,
    *,
    run_id: str = "tv-" + "a" * 32,
    generated_at: str = "2026-09-20T10:00:00Z",
) -> Path:
    run_dir = root / "outputs/training_validation" / run_id
    run_dir.mkdir(parents=True)
    for relative in required_evidence_files():
        path = run_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("evidence\n", encoding="utf-8")

    partition_counts = {
        "TRAIN": 80,
        "EVALUATION_HOLDOUT": 19,
        "INFERENCE_RESERVE": 1,
    }
    (run_dir / "partition_declaration.json").write_text(
        json.dumps(
            {
                "schema_version": "training-validation/v1",
                "run_id": run_id,
                "train_end": "2025-10",
                "holdout_end": "2025-11",
                "counts": partition_counts,
                "shares": {
                    "TRAIN": 0.8,
                    "EVALUATION_HOLDOUT": 0.19,
                    "INFERENCE_RESERVE": 0.01,
                },
                "reserve_exposure": "attested-unexposed",
            }
        ),
        encoding="utf-8",
    )
    _write_csv(
        run_dir / "partition_membership.csv",
        [
            {
                "dataset_id": "fixture",
                "row_id": f"row-{index}",
                "source_position": index,
                "period": "2025-10" if index < 80 else "2025-11" if index < 99 else "2025-12",
                "partition": "TRAIN" if index < 80 else "EVALUATION_HOLDOUT" if index < 99 else "INFERENCE_RESERVE",
            }
            for index in range(100)
        ],
    )
    _write_csv(
        run_dir / "monthly_row_counts.csv",
        [
            {"dataset_id": "fixture", "period": f"2025-{month:02d}", "partition": "TRAIN", "row_count": 8, "evidence_ref": f"partition_membership.csv#period=2025-{month:02d}"}
            for month in range(1, 11)
        ]
        + [
            {"dataset_id": "fixture", "period": "2025-11", "partition": "EVALUATION_HOLDOUT", "row_count": 19, "evidence_ref": "partition_membership.csv#period=2025-11"},
            {"dataset_id": "fixture", "period": "2025-12", "partition": "INFERENCE_RESERVE", "row_count": 1, "evidence_ref": "partition_membership.csv#period=2025-12"},
        ],
    )
    fold_rows = []
    for ordinal in range(1, 6):
        fold_rows.append(
            {
                "fold_id": f"outer-{ordinal}",
                "scope": "outer",
                "parent_fold_id": None,
                "search_id": None,
                "ordinal": ordinal,
                "parent_population_rows": 80,
                "train_period_min": "2025-01",
                "train_period_max": f"2025-{ordinal + 4:02d}",
                "validation_period_min": f"2025-{ordinal + 5:02d}",
                "train_rows": 8 * (ordinal + 4),
                "validation_rows": 8,
                "not_used_yet_rows": 80 - 8 * (ordinal + 5),
                "added_train_rows": None if ordinal == 1 else 8,
                "holdout_rows_excluded": 19,
                "reserve_rows_excluded": 1,
                "row_overlap_count": 0,
                "shared_month_count": 0,
                "chronological_order_ok": True,
                "expanding_history_ok": None if ordinal == 1 else True,
                "evidence_ref": f"fold_membership.csv#fold_id=outer-{ordinal}",
            }
        )
    for parent_index, parent in enumerate(
        ["outer-1", "outer-2", "outer-3", "outer-4", "outer-5", "final-train"]
    ):
        for ordinal in range(1, 4):
            fold_rows.append(
                {
                    "fold_id": f"{parent}:inner-{ordinal}",
                    "scope": "inner",
                    "parent_fold_id": parent,
                    "search_id": f"rf-{parent}",
                    "ordinal": ordinal,
                    "parent_population_rows": 40 + parent_index * 8,
                    "train_period_min": "2025-01",
                    "train_period_max": f"2025-{ordinal + 1:02d}",
                    "validation_period_min": f"2025-{ordinal + 2:02d}",
                    "train_rows": 8 * (ordinal + 1),
                    "validation_rows": 8,
                    "not_used_yet_rows": max(0, 24 + parent_index * 8 - ordinal * 8),
                    "added_train_rows": None if ordinal == 1 else 8,
                    "holdout_rows_excluded": 19,
                    "reserve_rows_excluded": 1,
                    "row_overlap_count": 0,
                    "shared_month_count": 0,
                    "chronological_order_ok": True,
                    "expanding_history_ok": None if ordinal == 1 else True,
                    "evidence_ref": f"fold_membership.csv#fold_id={parent}:inner-{ordinal}",
                }
            )
    _write_csv(run_dir / "fold_summary.csv", fold_rows)
    _write_csv(
        run_dir / "fold_membership.csv",
        [{"fold_id": "outer-1", "scope": "outer", "parent_fold_id": None, "search_id": None, "role": "train", "row_id": "row-0", "membership_hash": "hash"}],
    )
    (run_dir / "fold_explanation.md").write_text(
        "# Fold explanation\nOuter fold 3 uses 56 training rows and 8 validation rows.\n",
        encoding="utf-8",
    )

    models = [
        "Dummy Median",
        "Linear Regression",
        "Ridge Regression",
        "Random Forest",
        "Gradient Boosting",
    ]
    summary_rows = []
    fold_metric_rows = []
    diagnostic_rows = []
    for model_index, model in enumerate(models):
        summary_rows.append(
            {
                "model": model,
                "configuration_id": f"cfg-{model_index}",
                "valid_fold_count": 5,
                "failed_fold_count": 0,
                "cv_mae_mean_usd": 20000 - model_index * 1000,
                "cv_mae_sd_usd": 1000 + model_index,
                "cv_rmse_mean_usd": 24000 - model_index * 900,
                "cv_medae_mean_usd": 18000 - model_index * 800,
                "cv_r2_mean": model_index / 10,
                "worst_fold_id": "outer-3",
                "total_cv_fit_wall_s": 0.1 + model_index,
            }
        )
        for fold_index in range(1, 6):
            fold_metric_rows.append(
                {
                    "model": model,
                    "configuration_id": f"cfg-{model_index}",
                    "fold_id": f"outer-{fold_index}",
                    "fold_ordinal": fold_index,
                    "train_rows": 8 * (fold_index + 4),
                    "validation_rows": 8,
                    "train_MAE": 15000 - model_index * 500,
                    "validation_MAE": 21000 - model_index * 1000 + fold_index,
                    "validation_RMSE": 25000 - model_index * 900,
                    "validation_MedAE": 19000 - model_index * 700,
                    "validation_R2": model_index / 10,
                    "fit_wall_s": 0.01 + model_index / 10,
                    "fit_cpu_s": 0.01,
                    "train_predict_wall_s": 0.001,
                    "validation_predict_wall_s": 0.001,
                }
            )
            diagnostic_rows.append(
                {"model": model, "fold_id": f"outer-{fold_index}", "fold_indication": "baseline" if model == "Dummy Median" else "good_fit_indication", "aggregate_indication": "baseline" if model == "Dummy Median" else "good_fit_indication"}
            )
    _write_csv(run_dir / "candidate_summary.csv", summary_rows)
    _write_csv(run_dir / "candidate_fold_metrics.csv", fold_metric_rows)
    _write_csv(run_dir / "fit_diagnostics.csv", diagnostic_rows)
    _write_csv(run_dir / "runtime.csv", [{"model": "Linear Regression", "fold_id": "outer-1", "fit_wall_s": 0.1}])
    _write_csv(run_dir / "runtime_samples.csv", [{"model_id": "final_full", "fold_id": "final_full", "batch_size": 1, "repeat": 1, "wall_ns": 1000, "cpu_ns": 900}])
    _write_csv(run_dir / "runtime_summary.csv", [{"model_id": "final_full", "batch_size": 1, "sample_count": 30, "p50_ms": 1.0, "p90_ms": 2.0, "p95_ms": 2.5}])
    _write_csv(run_dir / "performance_comparison.csv", [{"model": "Linear Regression", "pareto": True, "mean_residual_usd": -10.0}])
    _write_csv(run_dir / "accuracy_runtime_tradeoff.csv", [{"model": "Linear Regression", "cv_mae_mean_usd": 19000, "total_cv_fit_wall_s": 1.1, "pareto": True}])
    _write_csv(run_dir / "oof_predictions.csv", [{"model": "Linear Regression", "fold_id": "outer-1", "row_id": "row-40", "partition": "OUTER_VALIDATION", "actual": 60000, "predicted": 59000, "residual": 1000, "absolute_error": 1000}])
    _write_csv(run_dir / "ablation_metrics.csv", [{"removed_family": "skills", "fold_id": "outer-1", "validation_MAE": 20000, "mae_delta_vs_full": 500}])
    _write_csv(run_dir / "fold_importance.csv", [{"model": "Random Forest", "fold_id": "outer-1", "family": "skills", "importance": 0.2}])
    _write_csv(run_dir / "importance_drift.csv", [{"previous_fold_id": "outer-1", "fold_id": "outer-2", "half_l1_drift": 0.1}])
    (run_dir / "family_selection.json").write_text(json.dumps({"lowest_mae_model": "Gradient Boosting", "selected_family": "Linear Regression", "overlap_models": ["Linear Regression", "Gradient Boosting"], "selection_method": "overlap_simplicity"}), encoding="utf-8")

    tuning_rows = [
        {
            "search_id": f"rf-{context}",
            "method_version": "gridsearchcv-temporal/v1",
            "stage": "grid-search",
            "slot": 1,
            "trial_ordinal": 1,
            "trial_id": f"rf-{context}:candidate-1",
            "configuration_id": "rf-cfg",
            "params": "{n_estimators: 50, max_depth: 10}",
            "n_estimators": 50,
            "max_depth": 10,
            "min_samples_leaf": 2,
            "max_features": 0.8,
            "seed": 42,
            "fold_ids": "inner-1|inner-2|inner-3",
            "inner_fold_ids": "inner-1|inner-2|inner-3",
            "fold_count": 3,
            "fold_mae": "[16500, 17000, 17500]",
            "fold_r2": "[0.4, 0.5, 0.6]",
            "mae_mean": 17000,
            "mae_sd": 500,
            "r2_mean": 0.5,
            "r2_sd": 0.1,
            "mean_fit_time_s": 0.1,
            "std_fit_time_s": 0.01,
            "ranking_metric": "MAE",
            "ranking_direction": "lower",
            "scoring": "neg_mean_absolute_error",
            "rank": 1,
            "status": "completed",
            "failure_reason": None,
            "evidence_ref": f"tuning_trials.csv#trial_id=rf-{context}:candidate-1",
        }
        for context in ["outer-1", "outer-2", "outer-3", "outer-4", "outer-5", "final-train"]
    ]
    _write_csv(run_dir / "tuning_trials.csv", tuning_rows)
    (run_dir / "tuning_summary.json").write_text(json.dumps({context: {"status": "completed", "reason": None, "winner": {"n_estimators": 50}, "actual_fit_count": 3} for context in ["outer-1", "outer-2", "outer-3", "outer-4", "outer-5", "final-train"]}), encoding="utf-8")
    _write_csv(
        run_dir / "variant_fold_metrics.csv",
        [
            {"fold_id": f"outer-{fold}", "variant": variant, "features": "all" if variant == "full" else "job_category|years_of_experience", "params": "{}", "validation_MAE": 15000 + fold + (1000 if variant == "top2" else 0), "validation_RMSE": 18000, "validation_MedAE": 14000, "validation_R2": 0.5}
            for fold in range(1, 6)
            for variant in ["full", "top2"]
        ],
    )
    (run_dir / "evaluation_declaration.json").write_text(json.dumps({"run_id": run_id, "selected_family": "Linear Regression", "final_variants": ["final_full", "final_top2"], "holdout_id": "holdout-fixture"}), encoding="utf-8")
    roles = models + ["final_full", "final_top2"]
    _write_csv(
        run_dir / "holdout_metrics.csv",
        [{"model_role_id": role, "partition": "EVALUATION_HOLDOUT", "rows": 19, "MAE": 14000 + index * 100, "RMSE": 18000 + index * 100, "MedAE": 13000 + index * 100, "R2": 0.5 - index / 100} for index, role in enumerate(roles)],
    )
    _write_csv(
        run_dir / "holdout_predictions.csv",
        [{"model_role_id": role, "row_id": "row-80", "partition": "EVALUATION_HOLDOUT", "actual": 70000, "predicted": 69000, "residual": 1000, "absolute_error": 1000, "job_category": "Data Science", "country": "Vietnam", "years_of_experience": 3.0} for role in roles],
    )
    _write_csv(run_dir / "encoded_importance.csv", [{"model_role_id": "final_full", "encoded_feature": "num__years_of_experience", "raw_feature": "years_of_experience", "method": "absolute_coefficient", "importance": 0.8}])
    _write_csv(run_dir / "permutation_importance.csv", [{"kind": "raw_feature", "name": "years_of_experience", "repeat": repeat, "scoring": "mae_increase_usd", "baseline_mae_usd": 14000, "permuted_mae_usd": 16000, "mae_increase": 2000, "seed": 42} for repeat in range(1, 13)])
    _write_csv(run_dir / "subgroup_metrics.csv", [{"model_role_id": "final_full", "subgroup": "country", "value": "Vietnam", "support": 5, "small_sample": True, "MAE": 15000, "RMSE": 18000, "MedAE": 14000, "R2": 0.4}])
    (run_dir / "uncertainty.json").write_text(json.dumps([{"model_role_id": "final_full", "q90_absolute_error_usd": 22000, "quantile": 0.9, "quantile_method": "linear", "same_population_coverage": 0.9, "basis": "same_holdout_descriptive_not_calibrated", "rows": 19}, {"model_role_id": "final_top2", "q90_absolute_error_usd": 25000, "quantile": 0.9, "quantile_method": "linear", "same_population_coverage": 0.9, "basis": "same_holdout_descriptive_not_calibrated", "rows": 19}]), encoding="utf-8")
    (run_dir / "operational_assessment.json").write_text(json.dumps({"operational_status": "not_assessed", "deployment_review_eligible": False}), encoding="utf-8")
    (run_dir / "conclusion.json").write_text(json.dumps({"execution_status": "complete", "scientific_outcome": {"status": "Good"}, "selected_model_id": "Linear Regression", "next_action": "Human review"}), encoding="utf-8")
    conclusions = []
    for role in roles:
        conclusions.append({"model_role_id": role, "status": "available", "finding": f"Finding for {role}", "decision": "selected_family" if role == "Linear Regression" else "comparison_only", "limitation": "Fixture limitation", "next_action": "Review evidence", "evidence_refs": [f"candidate_summary.csv#model={role}"]})
    (run_dir / "model_conclusions.json").write_text(json.dumps(conclusions), encoding="utf-8")
    (run_dir / "agent_summary.json").write_text(json.dumps({"run_id": run_id, "lowest_cv_mae_model_id": "Gradient Boosting", "selected_family_id": "Linear Regression", "fastest_fit_model_id": "Dummy Median", "folds": fold_rows[:5], "model_conclusions": conclusions}), encoding="utf-8")
    (run_dir / "ui_summary.json").write_text(json.dumps({"run_id": run_id, "page04": {}, "page05": {}}), encoding="utf-8")
    (run_dir / "metric_catalog.json").write_text(json.dumps({"mae_usd": {"unit": "USD/year"}}), encoding="utf-8")
    (run_dir / "training.log").write_text("Training validation completed.\nSelected family: Linear Regression.\nInference reserve was not evaluated.\n", encoding="utf-8")
    (run_dir / "events.jsonl").write_text(json.dumps({"schema_version": "training-validation/v1", "run_id": run_id, "sequence": 1, "timestamp": generated_at, "event_type": "run_started", "status": "started", "message": "Authorized training validation started."}) + "\n" + json.dumps({"schema_version": "training-validation/v1", "run_id": run_id, "sequence": 2, "timestamp": generated_at, "event_type": "operation_completed", "status": "completed", "message": "Authorized training validation completed."}) + "\n", encoding="utf-8")
    (run_dir / "report.md").write_text("# Training Validation Report\n\nDetailed fixture report.\n", encoding="utf-8")

    files = []
    for relative in sorted(required_evidence_files()):
        path = run_dir / relative
        files.append({"path": f"outputs/training_validation/{run_id}/{relative}", "sha256": sha256_file(path), "size": path.stat().st_size})
    manifest = {
        "schema_version": "training-validation/v1",
        "training_method_version": "gridsearchcv-temporal/v1",
        "run_id": run_id,
        "execution_status": "complete",
        "generated_at": generated_at,
        "experiment_sha256": "f" * 64,
        "selected_family": "Linear Regression",
        "scientific_outcome": {"status": "Good"},
        "files": files,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return run_dir


def test_training_events_sort_by_operation_status_then_model() -> None:
    events = pd.DataFrame(
        [
            {"operation": "tune", "status": "completed", "model": "B", "sequence": 4},
            {"operation": "score", "status": "completed", "model": "A", "sequence": 2},
            {"operation": "tune", "status": "started", "model": "B", "sequence": 3},
            {"operation": "tune", "status": "started", "model": "A", "sequence": 1},
        ]
    )

    result = sort_training_events(events)

    assert list(zip(result["operation"], result["status"], result["model"])) == [
        ("score", "completed", "A"),
        ("tune", "started", "A"),
        ("tune", "started", "B"),
        ("tune", "completed", "B"),
    ]


def test_discovery_reports_missing_and_ignores_staging_and_ledger(tmp_path: Path) -> None:
    missing = discover_latest_training_validation(tmp_path)
    assert missing["available"] is False
    assert missing["reason_code"] == "PACK_ROOT_MISSING"
    root = tmp_path / "outputs/training_validation"
    (root / (".tv-" + "b" * 32 + ".staging")).mkdir(parents=True)
    (root / "holdout_access").mkdir()
    (root / ("tv-" + "A" * 32)).mkdir()
    (root / "tv-too-short").mkdir()
    empty = discover_latest_training_validation(tmp_path)
    assert empty["reason_code"] == "NO_FINAL_PACK"
    assert empty["fallback_used"] is False


def test_discovery_selects_latest_valid_utc_pack_and_breaks_ties_by_run_id(tmp_path: Path) -> None:
    first = "tv-" + "1" * 32
    second = "tv-" + "2" * 32
    build_complete_pack(tmp_path, run_id=first, generated_at="2026-09-20T10:00:00Z")
    build_complete_pack(tmp_path, run_id=second, generated_at="2026-09-20T11:00:00Z")
    selected = discover_latest_training_validation(tmp_path)
    assert selected["available"] is True
    assert selected["run_id"] == second

    manifests = [tmp_path / "outputs/training_validation" / run / "manifest.json" for run in (first, second)]
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text())
        manifest["generated_at"] = "2026-09-20T12:00:00Z"
        manifest_path.write_text(json.dumps(manifest))
    selected = discover_latest_training_validation(tmp_path)
    assert selected["run_id"] == second


def test_discovery_invalidates_old_manual_search_pack(tmp_path: Path) -> None:
    run_id = "tv-" + "9" * 32
    run_dir = build_complete_pack(tmp_path, run_id=run_id)
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["training_method_version"] = "manual-stepwise/v1"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = discover_latest_training_validation(tmp_path)

    assert result["available"] is False
    assert result["reason_code"] == "NO_VALID_PACK"
    assert result["invalid_candidate_count"] == 1


def test_newer_corrupt_pack_does_not_hide_older_valid_pack(tmp_path: Path) -> None:
    valid = "tv-" + "3" * 32
    corrupt = "tv-" + "4" * 32
    build_complete_pack(tmp_path, run_id=valid, generated_at="2026-09-20T10:00:00Z")
    corrupt_dir = build_complete_pack(tmp_path, run_id=corrupt, generated_at="2026-09-20T12:00:00Z")
    (corrupt_dir / "fold_summary.csv").write_text("corrupt\n")
    selected = discover_latest_training_validation(tmp_path)
    assert selected["run_id"] == valid
    assert selected["invalid_candidate_count"] == 1
    assert selected["invalid_candidates"][0]["run_id"] == corrupt
    assert "checksum" not in selected["invalid_candidates"][0].get("raw_error", "")


def test_discovery_rejects_non_utc_timestamp_and_symlink(tmp_path: Path) -> None:
    run_id = "tv-" + "5" * 32
    run_dir = build_complete_pack(tmp_path, run_id=run_id, generated_at="2026-09-20 10:00")
    result = discover_latest_training_validation(tmp_path)
    assert result["available"] is False
    assert result["reason_code"] == "NO_VALID_PACK"
    run_dir.rename(tmp_path / "real-pack")
    (tmp_path / "outputs/training_validation" / run_id).symlink_to(tmp_path / "real-pack", target_is_directory=True)
    result = discover_latest_training_validation(tmp_path)
    assert result["reason_code"] == "NO_VALID_PACK"


def test_discovery_fails_closed_above_run_limit(tmp_path: Path) -> None:
    root = tmp_path / "outputs/training_validation"
    for index in range(101):
        (root / f"tv-{index:032x}").mkdir(parents=True)
    result = discover_latest_training_validation(tmp_path)
    assert result["available"] is False
    assert result["reason_code"] == "PACK_LIMIT_EXCEEDED"


def test_view_reconciles_folds_models_transcript_and_downloads(tmp_path: Path) -> None:
    run_id = "tv-" + "6" * 32
    build_complete_pack(tmp_path, run_id=run_id)
    view = load_training_validation_view(tmp_path)
    assert view["available"] is True
    assert len(view["page04"]["outer_folds"]) == 5
    assert len(view["page04"]["inner_folds"]) == 18
    outer3 = view["page04"]["outer_folds"].query("fold_id == 'outer-3'").iloc[0]
    assert outer3["train_rows"] == 56
    assert outer3["validation_rows"] == 8
    assert outer3["validation_period_min"] == "2025-08"
    assert len(view["page04"]["candidate_folds"]) == 25
    assert len(view["page04"]["conclusions"]) == 5
    assert len(view["page05"]["variant_folds"]) == 10
    assert len(view["page05"]["holdout_metrics"]) == 7
    assert len(view["page05"]["conclusions"]) == 2
    transcript = view["transcript"]
    assert (transcript["record_kind"] == "evidence_derived").all()
    assert transcript["source_file"].notna().all()
    assert len(transcript.query("stage == 'partition_month'")) == 12
    assert len(transcript.query("stage == 'candidate_fold'")) == 25
    assert len(transcript.query("stage == 'outer_fold'")) == 5
    assert len(transcript.query("stage == 'inner_fold'")) == 18
    assert len(transcript.query("stage == 'tuning_trial'")) == 12
    assert len(transcript.query("stage == 'variant_fold'")) == 10
    assert len(transcript.query("stage == 'final_fit'")) == 2
    assert len(transcript.query("stage == 'holdout'")) == 7
    assert len(transcript.query("stage == 'holdout_residual'")) == 7
    assert len(transcript.query("stage == 'conclusion'")) == 7
    assert len(transcript.query("stage == 'operational'")) == 1
    assert list(transcript["sequence"]) == list(range(1, len(transcript) + 1))
    downloads = {item["filename"]: item for item in view["downloads"]}
    for filename in ["training.log", "events.jsonl", "report.md", "agent_summary.json", "model_conclusions.json", "training-validation-transcript.csv"]:
        assert filename in downloads
        assert downloads[filename]["content"]
    run_dir = tmp_path / "outputs/training_validation" / run_id
    for filename in [
        "training.log",
        "events.jsonl",
        "report.md",
        "agent_summary.json",
        "model_conclusions.json",
    ]:
        assert downloads[filename]["content"] == (run_dir / filename).read_bytes()
        assert downloads[filename]["sha256"] == sha256_file(run_dir / filename)
    assert len(pd.read_csv(pd.io.common.BytesIO(downloads["training-validation-transcript.csv"]["content"]))) == len(transcript)


def test_view_accepts_explicitly_skipped_tuning_contexts(tmp_path: Path) -> None:
    run_id = "tv-" + "8" * 32
    run_dir = build_complete_pack(tmp_path, run_id=run_id)
    pd.DataFrame(columns=["search_id", "status"]).to_csv(
        run_dir / "tuning_trials.csv", index=False
    )
    summary = {
        context: {
            "status": "skipped",
            "reason": "selected_family_is_not_random_forest",
            "winner": None,
            "actual_fit_count": 0,
        }
        for context in [
            "outer-1",
            "outer-2",
            "outer-3",
            "outer-4",
            "outer-5",
            "final-train",
        ]
    }
    (run_dir / "tuning_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    manifest = json.loads((run_dir / "manifest.json").read_text())
    for filename in ("tuning_trials.csv", "tuning_summary.json"):
        entry = next(item for item in manifest["files"] if item["path"].endswith(filename))
        entry["sha256"] = sha256_file(run_dir / filename)
        entry["size"] = (run_dir / filename).stat().st_size
    (run_dir / "manifest.json").write_text(json.dumps(manifest))

    view = load_training_validation_view(tmp_path)
    contexts = view["transcript"].query("stage == 'tuning_trial'")
    assert len(contexts) == 6
    assert set(contexts["status"]) == {"skipped"}
    assert set(contexts["reason"]) == {"selected_family_is_not_random_forest"}


def test_deprecated_pack_panels_are_not_rendered_by_pages(tmp_path: Path) -> None:
    build_complete_pack(tmp_path, run_id="tv-" + "9" * 32)
    script = (
        "from pathlib import Path\n"
        "import streamlit as st\n"
        "from pages import page04_model_comparison as page\n"
        f"page.render(st, Path({str(tmp_path)!r}), 'admin')\n"
    )

    app = AppTest.from_string(script, default_timeout=30).run()

    assert not app.exception
    assert "Training & validation evidence" not in [item.value for item in app.subheader]
    assert not any("Latest training validation:" in item.label for item in app.expander)


def test_optional_sections_remain_local_when_values_are_unavailable(tmp_path: Path) -> None:
    run_id = "tv-" + "b" * 32
    run_dir = build_complete_pack(tmp_path, run_id=run_id)
    candidates = pd.read_csv(run_dir / "candidate_summary.csv")
    candidates["cv_r2_mean"] = None
    candidates.to_csv(run_dir / "candidate_summary.csv", index=False)
    pd.DataFrame(
        columns=[
            "model_role_id",
            "encoded_feature",
            "raw_feature",
            "method",
            "importance",
        ]
    ).to_csv(run_dir / "encoded_importance.csv", index=False)
    pd.DataFrame(
        columns=["model_id", "batch_size", "sample_count", "p50_ms", "p90_ms", "p95_ms"]
    ).to_csv(run_dir / "runtime_summary.csv", index=False)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    for filename in (
        "candidate_summary.csv",
        "encoded_importance.csv",
        "runtime_summary.csv",
    ):
        entry = next(item for item in manifest["files"] if item["path"].endswith(filename))
        entry["sha256"] = sha256_file(run_dir / filename)
        entry["size"] = (run_dir / filename).stat().st_size
    (run_dir / "manifest.json").write_text(json.dumps(manifest))

    view = load_training_validation_view(tmp_path)
    assert view["available"] is True
    assert view["page04"]["candidate_summary"]["cv_r2_mean"].isna().all()
    assert view["page04"]["runtime_summary"].empty
    assert view["page05"]["encoded_importance"].empty
    assert bool(view["page05"]["subgroups"].iloc[0]["small_sample"]) is True
    assert len(view["page05"]["holdout_metrics"]) == 7


@pytest.mark.parametrize(
    ("module_name", "supplemental_error"),
    [
        ("page04_model_comparison", "Supplemental model evidence is unavailable"),
        ("page05_best_model", "Supplemental diagnostic evidence is unavailable"),
    ],
)
def test_training_and_supplemental_sources_cover_remaining_state_matrix(
    tmp_path: Path, module_name: str, supplemental_error: str
) -> None:
    project = Path(__file__).resolve().parents[1]
    source = project / "outputs"
    destination = tmp_path / "available/outputs"
    destination.parent.mkdir(parents=True)
    shutil.copytree(source, destination)
    shutil.copytree(project / "artifacts", tmp_path / "available/artifacts")
    build_complete_pack(tmp_path / "available", run_id="tv-" + "c" * 32)
    available_script = (
        "from pathlib import Path\n"
        "import streamlit as st\n"
        f"from pages import {module_name} as page\n"
        f"page.render(st, Path({str(tmp_path / 'available')!r}), 'admin')\n"
    )
    available = AppTest.from_string(available_script, default_timeout=30).run()
    assert not available.exception
    assert not any(supplemental_error in item.value for item in available.error)
    assert available.get("plotly_chart")

    missing_root = tmp_path / "both-missing"
    missing_root.mkdir()
    missing_script = (
        "from pathlib import Path\n"
        "import streamlit as st\n"
        f"from pages import {module_name} as page\n"
        f"page.render(st, Path({str(missing_root)!r}), 'admin')\n"
    )
    missing = AppTest.from_string(missing_script, default_timeout=30).run()
    assert not missing.exception
    assert any("Historical training audit unavailable" in item.value for item in missing.info)
    assert any(supplemental_error in item.value for item in missing.error)


def test_view_rejects_reserve_metrics_and_missing_candidate_role(tmp_path: Path) -> None:
    run_id = "tv-" + "7" * 32
    run_dir = build_complete_pack(tmp_path, run_id=run_id)
    metrics = pd.read_csv(run_dir / "holdout_metrics.csv")
    metrics.loc[0, "partition"] = "INFERENCE_RESERVE"
    metrics.to_csv(run_dir / "holdout_metrics.csv", index=False)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    entry = next(item for item in manifest["files"] if item["path"].endswith("holdout_metrics.csv"))
    entry["sha256"] = sha256_file(run_dir / "holdout_metrics.csv")
    entry["size"] = (run_dir / "holdout_metrics.csv").stat().st_size
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="reserve|EVALUATION_HOLDOUT"):
        load_training_validation_view(tmp_path)
