from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

from pages.model_evidence import (
    actual_predicted_figure,
    candidate_comparison_figure,
    importance_figure,
)

from .core import (
    MODEL_FEATURES,
    TARGET,
    candidate_models,
    temporal_cv_splits,
)
from .scenario_policy import build_scenario_policy
from .ui_evidence_io import (
    EvidenceContractError,
    atomic_publish,
    file_entry,
    load_current_evidence,
    sha256_file,
    write_json,
)
from .ui_evidence_training import (
    evaluate_frozen_models,
    evaluate_test_models,
    permutation_evidence,
    score_variant,
    select_benchmark_offsets,
)

TOP2_FEATURES = ["job_category", "years_of_experience"]
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _required(workspace: Path, relative: str) -> Path:
    path = workspace / relative
    if not path.is_file():
        raise EvidenceContractError(f"required evidence missing: {relative}")
    return path


def _source_paths(workspace: Path) -> dict[str, Path]:
    return {
        "development": _required(workspace, "outputs/02_data_ready_for_ml/development_raw.csv"),
        "historical_test": _required(workspace, "outputs/02_data_ready_for_ml/locked_test_raw.csv"),
        "split_summary": _required(
            workspace, "outputs/02_data_ready_for_ml/temporal_split_summary.json"
        ),
        "metadata": _required(workspace, "artifacts/metadata.json"),
        "feature_contract": _required(workspace, "artifacts/feature_contract.json"),
        "full_bundle": _required(workspace, "artifacts/model_bundle.joblib"),
    }


def _dependencies(workspace: Path, sources: dict[str, Path]) -> tuple[str, list[dict[str, Any]]]:
    producer_files = [
        PROJECT_ROOT / "src/ai_job_market/core.py",
        PROJECT_ROOT / "src/ai_job_market/ui_evidence.py",
        PROJECT_ROOT / "src/ai_job_market/ui_evidence_training.py",
        PROJECT_ROOT / "src/ai_job_market/ui_evidence_io.py",
        PROJECT_ROOT / "src/ai_job_market/scenario_policy.py",
        PROJECT_ROOT / "src/pages/model_evidence.py",
        PROJECT_ROOT / "config/project.yaml",
        PROJECT_ROOT / "uv.lock",
    ]
    records: list[dict[str, Any]] = []
    for role, path in sources.items():
        records.append(
            {
                "role": role,
                "path": path.resolve().relative_to(workspace).as_posix(),
                "sha256": sha256_file(path),
                "size": path.stat().st_size,
            }
        )
    for path in producer_files:
        if path.is_file():
            records.append(
                {
                    "role": "producer",
                    "path": path.resolve().relative_to(PROJECT_ROOT).as_posix(),
                    "sha256": sha256_file(path),
                    "size": path.stat().st_size,
                }
            )
    raw = json.dumps(records, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest(), records


def check_workspace(workspace: Path | str) -> dict[str, Any]:
    root = Path(workspace).resolve()
    try:
        sources = _source_paths(root)
        dependency_digest, _ = _dependencies(root, sources)
        manifest = load_current_evidence(root)
        if manifest.get("dependency_digest") != dependency_digest:
            return {"valid": False, "message": "current evidence is stale for this workspace"}
        return {
            "valid": True,
            "message": "supplemental evidence is valid",
            "evidence_id": manifest["evidence_id"],
        }
    except (EvidenceContractError, FileNotFoundError, json.JSONDecodeError) as exc:
        return {"valid": False, "message": str(exc)}


def _save_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _model_params(estimator: Any) -> dict[str, Any]:
    return {
        key: value
        for key, value in estimator.get_params(deep=False).items()
        if isinstance(value, (str, int, float, bool, type(None)))
    }


def _importance_drift(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame(
            columns=[
                "model_id",
                "scope",
                "feature",
                "importance_mean",
                "importance_std",
                "importance_min",
                "importance_max",
                "folds",
                "importance_cv",
            ]
        )
    encoded = rows.rename(columns={"encoded_feature": "feature"}).copy()
    encoded["scope"] = "encoded"
    family = (
        rows.groupby(["model_id", "configuration_id", "fold_id", "raw_family"], as_index=False)
        .importance.sum()
        .rename(columns={"raw_family": "feature"})
    )
    family["scope"] = "raw_family"
    combined = pd.concat(
        [
            encoded[["model_id", "configuration_id", "fold_id", "scope", "feature", "importance"]],
            family,
        ],
        ignore_index=True,
    )
    drift = (
        combined.groupby(["model_id", "configuration_id", "scope", "feature"], as_index=False)
        .importance.agg(["mean", "std", "min", "max", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": "importance_mean",
                "std": "importance_std",
                "min": "importance_min",
                "max": "importance_max",
                "count": "folds",
            }
        )
    )
    drift["importance_std"] = drift["importance_std"].fillna(0.0)
    drift["importance_cv"] = drift.importance_std / drift.importance_mean.replace(0, np.nan)
    return drift


def generate(workspace: Path | str) -> dict[str, Any]:
    root = Path(workspace).resolve(strict=True)
    sources = _source_paths(root)
    dependency_digest, source_records = _dependencies(root, sources)
    evidence_id = f"ui-{dependency_digest[:16]}"
    try:
        current = load_current_evidence(root)
        if current.get("dependency_digest") == dependency_digest:
            return {"evidence_id": current["evidence_id"], "status": "reused"}
    except EvidenceContractError:
        pass

    output_dir = root / "outputs/ui_evidence" / evidence_id
    artifact_dir = root / "artifacts/ui_evidence" / evidence_id
    existing_manifest = output_dir / "manifest.json"
    if existing_manifest.is_file():
        manifest = json.loads(existing_manifest.read_text(encoding="utf-8"))
        atomic_publish(root, manifest)
        return {"evidence_id": evidence_id, "status": "reused"}
    if output_dir.exists() or artifact_dir.exists():
        shutil.rmtree(output_dir, ignore_errors=True)
        shutil.rmtree(artifact_dir, ignore_errors=True)
    output_dir.mkdir(parents=True)
    artifact_dir.mkdir(parents=True)

    dev = pd.read_csv(sources["development"])
    test = pd.read_csv(sources["historical_test"])
    metadata = json.loads(sources["metadata"].read_text(encoding="utf-8"))
    contract = json.loads(sources["feature_contract"].read_text(encoding="utf-8"))
    split_summary = json.loads(sources["split_summary"].read_text(encoding="utf-8"))
    if (
        contract.get("model_features") != MODEL_FEATURES
        or metadata.get("model_features") != MODEL_FEATURES
    ):
        raise EvidenceContractError("baseline full-feature contracts are incompatible")
    if TARGET not in dev or TARGET not in test:
        raise EvidenceContractError("prepared partitions are missing the salary target")
    if len(dev) != int(split_summary.get("development_rows", -1)) or len(test) != int(
        split_summary.get("locked_test_rows", -1)
    ):
        raise EvidenceContractError("prepared partition counts conflict with split summary")

    seed = 42
    dataset_dev = sha256_file(sources["development"])
    dataset_test = sha256_file(sources["historical_test"])
    policy = build_scenario_policy(dev, metadata, dataset_id=dataset_dev)
    policy_path = output_dir / "scenario_policy.json"
    write_json(policy_path, policy)

    full_bundle = joblib.load(sources["full_bundle"])
    if not hasattr(full_bundle, "named_steps") or "model" not in full_bundle.named_steps:
        raise EvidenceContractError("baseline full bundle is not a compatible sklearn pipeline")
    full_estimator = clone(full_bundle.named_steps["model"])
    if type(full_estimator).__name__ != "RandomForestRegressor":
        raise EvidenceContractError("Top-2 mode requires the selected full Random Forest settings")
    full_params = _model_params(full_estimator)
    splits = temporal_cv_splits(dev, int(metadata.get("temporal_cv_folds", 5)))

    declaration = {
        "schema_version": 1,
        "target": TARGET,
        "development_dataset_id": dataset_dev,
        "historical_test_dataset_id": dataset_test,
        "selection_policy": "frozen_diagnostic_only",
        "candidate_parameters": {
            name: _model_params(model) for name, model in candidate_models(seed).items()
        },
        "full_parameters": full_params,
        "full_features": MODEL_FEATURES,
        "top2_features": TOP2_FEATURES,
        "seed": seed,
        "folds": [
            {
                "fold_id": i,
                "train_offsets": tr.tolist(),
                "validation_offsets": va.tolist(),
                "validation_period": label,
            }
            for i, (tr, va, label) in enumerate(splits, 1)
        ],
        "historical_test_exposure": "already_scored; diagnostics cannot restore pristine status",
    }
    declaration_path = output_dir / "evaluation_declaration.json"
    write_json(declaration_path, declaration)

    membership_rows = []
    for fold_id, (train_idx, validation_idx, _) in enumerate(splits, 1):
        membership_rows.extend(
            {
                "fold_id": fold_id,
                "partition_role": "train",
                "record_id": f"development:{dataset_dev}:{int(i)}",
            }
            for i in train_idx
        )
        membership_rows.extend(
            {
                "fold_id": fold_id,
                "partition_role": "validation",
                "record_id": f"development:{dataset_dev}:{int(i)}",
            }
            for i in validation_idx
        )
    fold_membership = pd.DataFrame(membership_rows)

    candidate_fold, candidate_summary, rf_importance = evaluate_frozen_models(
        dev, MODEL_FEATURES, candidate_models(seed), splits, capture_importance=True
    )
    candidate_test, _ = evaluate_test_models(dev, test, MODEL_FEATURES, candidate_models(seed))

    variant_models = {"Full 13": full_estimator, "Top 2": clone(full_estimator)}
    variant_folds_parts = []
    variant_summaries = []
    for name, features, variant in [
        ("Full 13", MODEL_FEATURES, "full"),
        ("Top 2", TOP2_FEATURES, "top2"),
    ]:
        folds, summary, _ = evaluate_frozen_models(
            dev,
            features,
            {name: variant_models[name]},
            splits,
            capture_importance=False,
            feature_variant=variant,
        )
        variant_folds_parts.append(folds)
        summary["evaluation"] = "dev_cv_mean"
        summary = summary.rename(
            columns={
                "validation_MAE_mean": "MAE",
                "validation_RMSE_mean": "RMSE",
                "validation_R2_mean": "R2",
                "validation_MedAE_mean": "MedAE",
            }
        )
        summary["row_count"] = len(dev)
        variant_summaries.append(summary)
    variant_fold = pd.concat(variant_folds_parts, ignore_index=True)

    top2_pipeline = __import__(
        "ai_job_market.core", fromlist=["make_model_pipeline"]
    ).make_model_pipeline(clone(full_estimator), TOP2_FEATURES)
    top2_pipeline.fit(dev[TOP2_FEATURES], dev[TARGET])
    top2_model_path = artifact_dir / "top2_model.joblib"
    joblib.dump(top2_pipeline, top2_model_path)

    full_metrics, full_predictions = score_variant(
        "full:selected", full_bundle, test, MODEL_FEATURES
    )
    top2_metrics, top2_predictions = score_variant("top2:fixed", top2_pipeline, test, TOP2_FEATURES)
    variant_metrics = pd.concat(variant_summaries, ignore_index=True)
    variant_metrics = pd.concat(
        [variant_metrics, pd.DataFrame([full_metrics, top2_metrics])], ignore_index=True, sort=False
    )
    for frame in (full_predictions, top2_predictions):
        frame["record_id"] = frame.record_offset.map(
            lambda offset: f"historical_test:{dataset_test}:{offset}"
        )
    variant_predictions = pd.concat([full_predictions, top2_predictions], ignore_index=True)

    full_raw, full_encoded = permutation_evidence(
        "full:selected", full_bundle, test, MODEL_FEATURES, seed=seed
    )
    top2_raw, top2_encoded = permutation_evidence(
        "top2:fixed", top2_pipeline, test, TOP2_FEATURES, seed=seed
    )
    permutation = pd.concat([full_raw, top2_raw], ignore_index=True)
    encoded = pd.concat([full_encoded, top2_encoded], ignore_index=True)

    pairs = {(p["job_category"], p["job_title"]) for p in policy["pairs"]}
    eligible = pd.Series(
        [
            (str(row.job_category), str(row.job_title)) in pairs
            and str(row.job_title) in policy["titles"]
            and policy["titles"][str(row.job_title)]["min"]
            <= float(row.years_of_experience)
            <= policy["titles"][str(row.job_title)]["max"]
            for row in test.itertuples(index=False)
        ]
    )
    benchmark_offsets = select_benchmark_offsets(test, eligible, limit=3)
    full_lookup = full_predictions.set_index("record_offset")
    top2_lookup = top2_predictions.set_index("record_offset")
    benchmark_rows = []
    for offset in benchmark_offsets:
        source = test.iloc[offset].to_dict()
        row = {
            **source,
            "record_id": f"historical_test:{dataset_test}:{offset}",
            "record_offset": offset,
            "predicted_full_usd": float(full_lookup.loc[offset, "predicted_salary_usd"]),
            "absolute_error_full_usd": float(full_lookup.loc[offset, "absolute_error_usd"]),
            "predicted_top2_usd": float(top2_lookup.loc[offset, "predicted_salary_usd"]),
            "absolute_error_top2_usd": float(top2_lookup.loc[offset, "absolute_error_usd"]),
            "included_in_benchmark": True,
            "historically_exposed": True,
            "pristine": False,
            "strict_policy_eligible": True,
            "selection_rule": "first_strict_eligible_by_source_row_offset",
        }
        benchmark_rows.append(row)
    benchmarks = pd.DataFrame(benchmark_rows)

    top2_metadata = {
        "schema_version": 1,
        "evidence_id": evidence_id,
        "model_id": "top2:fixed",
        "model_name": "Random Forest (fixed Top-2)",
        "feature_order": TOP2_FEATURES,
        "estimator_parameters": full_params,
        "development_dataset_id": dataset_dev,
        "historical_test_dataset_id": dataset_test,
        "q90_abs_error_usd": top2_metrics["q90_abs_error_usd"],
        "q90_basis": top2_metrics["q90_basis"],
        "historical_test_metrics": {
            key: top2_metrics[key] for key in ["MAE", "RMSE", "R2", "MedAE", "coverage"]
        },
        "limitations": [
            "Fixed feature ablation; not independently tuned.",
            "Historical test error band is not guaranteed future or extrapolation coverage.",
        ],
    }
    top2_metadata_path = artifact_dir / "top2_metadata.json"
    write_json(top2_metadata_path, top2_metadata)

    tables = {
        "candidate_fold_metrics.csv": candidate_fold,
        "candidate_summary.csv": candidate_summary,
        "candidate_test_metrics.csv": candidate_test,
        "fold_membership.csv": fold_membership,
        "rf_fold_importance.csv": rf_importance,
        "rf_importance_drift.csv": _importance_drift(rf_importance),
        "variant_fold_metrics.csv": variant_fold,
        "variant_metrics.csv": variant_metrics,
        "variant_test_predictions.csv": variant_predictions,
        "variant_encoded_importance.csv": encoded,
        "variant_permutation_importance.csv": permutation,
        "benchmark_examples.csv": benchmarks,
    }
    for name, frame in tables.items():
        _save_csv(frame, output_dir / name)

    tuning_dir = output_dir / "tuning"
    tuning_sources = [
        "manual_tuning_step1_gridsearch.csv",
        "manual_tuning_step2_n_estimators.csv",
        "manual_tuning_step3_max_depth.csv",
        "manual_tuning_step4_min_samples_leaf.csv",
        "manual_tuning_step5_max_features.csv",
        "manual_tuning_4params_summary.csv",
    ]
    imported_tuning = []
    for name in tuning_sources:
        source = root / "outputs/05_best_model" / name
        if source.is_file():
            tuning_dir.mkdir(exist_ok=True)
            destination = tuning_dir / name
            shutil.copy2(source, destination)
            imported_tuning.append(destination)

    charts = output_dir / "charts"
    charts.mkdir()
    figures = {
        "candidate_comparison.html": candidate_comparison_figure(candidate_summary),
        "full_actual_predicted.html": actual_predicted_figure(variant_predictions, "full:selected"),
        "full_raw_importance.html": importance_figure(permutation, "full:selected"),
    }
    for name, figure in figures.items():
        figure.write_html(charts / name, include_plotlyjs="cdn", full_html=True)

    generated_paths = [policy_path, declaration_path, top2_model_path, top2_metadata_path]
    generated_paths.extend(output_dir / name for name in tables)
    generated_paths.extend(imported_tuning)
    generated_paths.extend(charts / name for name in figures)
    manifest = {
        "schema_version": 1,
        "evidence_id": evidence_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_run_id": metadata.get("run_id"),
        "target": TARGET,
        "full_features": MODEL_FEATURES,
        "top2_features": TOP2_FEATURES,
        "dependency_digest": dependency_digest,
        "source_files": source_records,
        "producer": {
            "seed": seed,
            "python": sys.version.split()[0],
            "versions": {
                name: importlib.metadata.version(name)
                for name in ["pandas", "numpy", "scikit-learn", "joblib", "plotly", "streamlit"]
            },
        },
        "datasets": {
            "development": {"id": dataset_dev, "rows": len(dev)},
            "historical_test": {
                "id": dataset_test,
                "rows": len(test),
                "period": split_summary.get("locked_test_period_label"),
            },
        },
        "models": {
            "full:selected": {
                "feature_order": MODEL_FEATURES,
                "parameters": full_params,
                "source": "baseline_bundle",
            },
            "top2:fixed": {
                "feature_order": TOP2_FEATURES,
                "parameters": full_params,
                "source": "supplemental_fixed_ablation",
            },
        },
        "components": {
            "comparison": {"status": "generated"},
            "variants": {"status": "generated"},
            "policy": {"status": "generated"},
            "benchmarks": {"status": "generated", "rows": len(benchmarks)},
            "serving": {"status": "generated"},
            "charts": {"status": "generated"},
            "tuning_history": {
                "status": "reused" if imported_tuning else "unavailable",
                "reason": "legacy snapshots imported with inherited non-nested DEV-search limitation"
                if imported_tuning
                else "legacy tuning snapshots missing",
            },
        },
        "files": [file_entry(root, path) for path in generated_paths],
        "limitations": [
            "Historical test rows were already scored and are not pristine.",
            "Top-2 uses frozen full-model settings and was not independently tuned.",
            "Imported tuning history is inherited non-nested DEV search ranked by R2.",
            "Feature reliance is not causal evidence or production fitness.",
        ],
    }
    atomic_publish(root, manifest)
    return {"evidence_id": evidence_id, "status": "generated", "components": manifest["components"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate supplemental evidence for model UI pages"
    )
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    if args.check:
        result = check_workspace(args.workspace)
        print(json.dumps(result, indent=2))
        return 0 if result["valid"] else 2
    try:
        print(json.dumps(generate(args.workspace), indent=2, default=str))
        return 0
    except Exception as exc:
        print(f"ui-evidence generation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
