"""Read-only Streamlit projections for immutable training-validation evidence."""

from __future__ import annotations

import json
import re
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ai_job_market.training_evidence_io import (
    EvidenceContractError,
    safe_workspace_path,
    sha256_file,
    validate_complete_pack,
)

_RUN_ID = re.compile(r"^tv-[0-9a-f]{32}$")
_UTC_TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$")
MAX_RUNS = 100
MAX_FILE_BYTES = 50 * 1024 * 1024
MAX_TABLE_ROWS = 100_000
MAX_PREVIEW_ROWS = 200
CANDIDATES = {
    "Dummy Median",
    "Linear Regression",
    "Ridge Regression",
    "Random Forest",
    "Gradient Boosting",
}
FINAL_ROLES = {"final_full", "final_top2"}
ALL_ROLES = CANDIDATES | FINAL_ROLES


def _unavailable(code: str, message: str, *, invalid: list[dict[str, str]] | None = None):
    return {
        "available": False,
        "reason_code": code,
        "message": message,
        "invalid_candidate_count": len(invalid or []),
        "invalid_candidates": (invalid or [])[:20],
        "inspect_command": (
            ".venv/bin/python training_validation.py inspect --workspace . "
            "--policy config/training_validation.json"
        ),
        "next_action": (
            "Obtain eligible future prepared data and exact partition/exposure approval "
            "before an offline run."
        ),
        "fallback_used": False,
    }


def _parse_utc(value: Any) -> datetime:
    text = str(value)
    if not _UTC_TIMESTAMP.fullmatch(text):
        raise ValueError("manifest generated_at must be a strict UTC timestamp")
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if parsed.utcoffset() is None:
        raise ValueError("manifest generated_at must include UTC")
    return parsed


def discover_latest_training_validation(workspace: Path | str) -> dict[str, Any]:
    try:
        root = Path(workspace).resolve(strict=True)
    except OSError:
        return _unavailable("ROOT_MISSING", "The selected evidence workspace is unavailable.")
    try:
        pack_root = safe_workspace_path(root, "outputs/training_validation")
    except EvidenceContractError:
        return _unavailable(
            "PACK_ROOT_UNSAFE", "The training-validation evidence root is not a safe local path."
        )
    if not pack_root.is_dir():
        return _unavailable(
            "PACK_ROOT_MISSING", "No offline training-validation pack has been published."
        )
    candidates = [
        path
        for path in pack_root.iterdir()
        if path.is_dir() and _RUN_ID.fullmatch(path.name)
    ]
    if len(candidates) > MAX_RUNS:
        return _unavailable(
            "PACK_LIMIT_EXCEEDED",
            f"Training-validation discovery exceeds the {MAX_RUNS}-run safety limit.",
        )
    if not candidates:
        return _unavailable("NO_FINAL_PACK", "No immutable final training-validation run exists.")
    valid: list[tuple[datetime, str, dict[str, Any], Path]] = []
    invalid: list[dict[str, str]] = []
    for directory in candidates:
        try:
            if directory.is_symlink():
                raise EvidenceContractError("run directory cannot be a symlink")
            manifest_path = safe_workspace_path(
                root, Path("outputs/training_validation") / directory.name / "manifest.json"
            )
            if not manifest_path.is_file() or manifest_path.stat().st_size > 1024 * 1024:
                raise EvidenceContractError("manifest is missing or oversized")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            generated = _parse_utc(manifest.get("generated_at"))
            validated = validate_complete_pack(root, directory.name)
            if validated.get("run_id") != directory.name:
                raise EvidenceContractError("run identity mismatch")
            valid.append((generated, directory.name, validated, manifest_path))
        except (EvidenceContractError, OSError, UnicodeError, ValueError, json.JSONDecodeError):
            invalid.append(
                {
                    "run_id": directory.name,
                    "reason_code": "PACK_INCOMPATIBLE",
                    "message": "The candidate pack failed integrity or schema validation.",
                }
            )
    if not valid:
        return _unavailable(
            "NO_VALID_PACK",
            "No complete training-validation pack passed the integrity and format checks.",
            invalid=invalid,
        )
    generated, run_id, manifest, manifest_path = max(
        valid, key=lambda item: (item[0], item[1])
    )
    return {
        "available": True,
        "run_id": run_id,
        "generated_at": generated.isoformat().replace("+00:00", "Z"),
        "manifest": manifest,
        "manifest_path": manifest_path.relative_to(root).as_posix(),
        "manifest_sha256": sha256_file(manifest_path),
        "invalid_candidate_count": len(invalid),
        "invalid_candidates": invalid[:20],
        "fallback_used": False,
    }


def _entry_map(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {Path(entry["path"]).name: entry for entry in manifest["files"]}


def _source_path(root: Path, entries: dict[str, dict[str, Any]], filename: str) -> Path:
    entry = entries.get(filename)
    if entry is None:
        raise ValueError(f"required evidence file is missing: {filename}")
    path = safe_workspace_path(root, entry["path"])
    if not path.is_file() or path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError(f"evidence file is unavailable or oversized: {filename}")
    return path


def _read_csv(root: Path, entries: dict[str, dict[str, Any]], filename: str) -> pd.DataFrame:
    frame = pd.read_csv(_source_path(root, entries, filename))
    if len(frame) > MAX_TABLE_ROWS:
        raise ValueError(f"evidence table exceeds row limit: {filename}")
    return frame


def _read_json(root: Path, entries: dict[str, dict[str, Any]], filename: str) -> Any:
    return json.loads(_source_path(root, entries, filename).read_text(encoding="utf-8"))


def _require_columns(frame: pd.DataFrame, filename: str, columns: set[str]) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{filename} is missing columns: {missing}")


def _transcript_row(stage: str, step: str, source: str, **values: Any) -> dict[str, Any]:
    return {
        "record_kind": "evidence_derived",
        "stage": stage,
        "step": step,
        "source_file": source,
        "evidence_ref": values.pop("evidence_ref", source),
        **values,
    }


def _build_transcript(
    *,
    partition: dict[str, Any],
    monthly: pd.DataFrame,
    folds: pd.DataFrame,
    candidate_folds: pd.DataFrame,
    selection: dict[str, Any],
    tuning_trials: pd.DataFrame,
    tuning_summary: dict[str, Any],
    variants: pd.DataFrame,
    evaluation_declaration: dict[str, Any],
    holdout: pd.DataFrame,
    prediction_summary: pd.DataFrame,
    encoded: pd.DataFrame,
    permutation: pd.DataFrame,
    subgroups: pd.DataFrame,
    uncertainty: list[dict[str, Any]],
    conclusions: list[dict[str, Any]],
    operational: dict[str, Any],
    outcome: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = [
        _transcript_row(
            "partition",
            "Approved chronological partition",
            "partition_declaration.json",
            partition="TRAIN / EVALUATION_HOLDOUT / INFERENCE_RESERVE",
            status="complete",
            metrics={"counts": partition.get("counts"), "shares": partition.get("shares")},
        )
    ]
    for _, row in monthly.sort_values(["period", "partition"]).iterrows():
        rows.append(
            _transcript_row(
                "partition_month",
                "Declare whole-month partition allocation",
                "monthly_row_counts.csv",
                partition=row["partition"],
                validation_periods=row["period"],
                validation_rows=int(row["row_count"]),
                status="complete",
                evidence_ref=str(row.get("evidence_ref") or f"monthly_row_counts.csv#period={row['period']}"),
            )
        )
    for _, row in folds.sort_values(["scope", "parent_fold_id", "ordinal"], na_position="first").iterrows():
        scope = str(row["scope"])
        rows.append(
            _transcript_row(
                "outer_fold" if scope == "outer" else "inner_fold",
                "Expanding monthly fold definition",
                "fold_summary.csv",
                fold_id=row["fold_id"],
                parent_fold_id=row.get("parent_fold_id"),
                search_id=row.get("search_id"),
                train_periods=f"{row['train_period_min']}..{row['train_period_max']}",
                validation_periods=row["validation_period_min"],
                parent_rows=int(row["parent_population_rows"]),
                train_rows=int(row["train_rows"]),
                validation_rows=int(row["validation_rows"]),
                not_used_yet_rows=int(row["not_used_yet_rows"]),
                added_train_rows=None if pd.isna(row.get("added_train_rows")) else int(row["added_train_rows"]),
                holdout_rows_excluded=int(row["holdout_rows_excluded"]),
                reserve_rows_excluded=int(row["reserve_rows_excluded"]),
                metrics={
                    "row_overlap_count": int(row.get("row_overlap_count", 0)),
                    "shared_month_count": int(row.get("shared_month_count", 0)),
                    "expanding_history_ok": None if pd.isna(row.get("expanding_history_ok")) else bool(row.get("expanding_history_ok")),
                },
                status="complete" if bool(row["chronological_order_ok"]) else "failed",
                evidence_ref=str(row.get("evidence_ref") or f"fold_summary.csv#fold_id={row['fold_id']}"),
            )
        )
    for _, row in candidate_folds.sort_values(["fold_ordinal", "model"]).iterrows():
        rows.append(
            _transcript_row(
                "candidate_fold",
                "Fit and score frozen candidate on outer fold",
                "candidate_fold_metrics.csv",
                model_role_id=row["model"],
                configuration_id=row["configuration_id"],
                fold_id=row["fold_id"],
                partition="OUTER_VALIDATION",
                train_rows=int(row["train_rows"]),
                validation_rows=int(row["validation_rows"]),
                metrics={
                    key: row.get(key)
                    for key in (
                        "train_MAE",
                        "validation_MAE",
                        "validation_RMSE",
                        "validation_MedAE",
                        "validation_R2",
                        "fit_wall_s",
                        "fit_cpu_s",
                        "train_predict_wall_s",
                        "validation_predict_wall_s",
                    )
                },
                status="completed",
                evidence_ref=f"candidate_fold_metrics.csv#model={row['model']}&fold_id={row['fold_id']}",
            )
        )
    rows.append(
        _transcript_row(
            "selection",
            "Freeze candidate-family selection",
            "family_selection.json",
            model_role_id=selection.get("selected_family"),
            status="completed",
            metrics={
                "lowest_mae_model": selection.get("lowest_mae_model"),
                "selected_family": selection.get("selected_family"),
                "selection_method": selection.get("selection_method"),
            },
        )
    )
    for context, summary in sorted(tuning_summary.items()):
        rows.append(
            _transcript_row(
                "tuning_trial",
                "Tuning context result",
                "tuning_summary.json",
                search_id=f"rf-{context}",
                parent_fold_id=context,
                parameters=summary.get("winner"),
                status=summary.get("status"),
                reason=summary.get("reason"),
                metrics={"actual_fit_count": summary.get("actual_fit_count")},
                evidence_ref=f"tuning_summary.json#{context}",
            )
        )
    if not tuning_trials.empty:
        required_trial_columns = {"search_id", "trial_ordinal", "trial_id", "status"}
        if required_trial_columns - set(tuning_trials.columns):
            raise ValueError("nonempty tuning trial evidence is missing identity columns")
        for _, row in tuning_trials.sort_values(["search_id", "trial_ordinal"]).iterrows():
            rows.append(
                _transcript_row(
                    "tuning_trial",
                    "Evaluate bounded tuning slot",
                    "tuning_trials.csv",
                    search_id=row.get("search_id"),
                    trial_id=row.get("trial_id"),
                    configuration_id=row.get("configuration_id"),
                    parameters=row.get("params"),
                    status=row.get("status"),
                    reason=row.get("reused_from"),
                    metrics={
                        "mae_mean": row.get("mae_mean"),
                        "mae_sd": row.get("mae_sd"),
                        "r2_mean": row.get("r2_mean"),
                        "rank": row.get("rank"),
                    },
                    evidence_ref=f"tuning_trials.csv#trial_id={row.get('trial_id')}",
                )
            )
    for _, row in variants.sort_values(["fold_id", "variant"]).iterrows():
        rows.append(
            _transcript_row(
                "variant_fold",
                "Evaluate matched Full/Top-2 variant",
                "variant_fold_metrics.csv",
                model_role_id=f"final_{row['variant']}",
                fold_id=row["fold_id"],
                partition="OUTER_VALIDATION",
                parameters=row.get("params"),
                metrics={key: row.get(key) for key in ("validation_MAE", "validation_RMSE", "validation_MedAE", "validation_R2")},
                status="completed",
                evidence_ref=f"variant_fold_metrics.csv#fold_id={row['fold_id']}&variant={row['variant']}",
            )
        )
    for role in evaluation_declaration.get("final_variants", []):
        rows.append(
            _transcript_row(
                "final_fit",
                "Fit frozen final variant on TRAIN only",
                "evaluation_declaration.json",
                model_role_id=role,
                partition="TRAIN",
                parameters={"selected_family": evaluation_declaration.get("selected_family")},
                status="completed",
                evidence_ref=f"evaluation_declaration.json#final_variants={role}",
            )
        )
    for _, row in holdout.sort_values("model_role_id").iterrows():
        rows.append(
            _transcript_row(
                "holdout",
                "Score frozen model role once on evaluation holdout",
                "holdout_metrics.csv",
                model_role_id=row["model_role_id"],
                partition=row["partition"],
                validation_rows=int(row["rows"]),
                metrics={key: row.get(key) for key in ("MAE", "RMSE", "MedAE", "R2")},
                status="completed",
                evidence_ref=f"holdout_metrics.csv#model_role_id={row['model_role_id']}",
            )
        )
    for _, row in prediction_summary.iterrows():
        rows.append(
            _transcript_row(
                "holdout_residual",
                "Aggregate holdout residual and absolute-error evidence",
                "holdout_predictions.csv",
                model_role_id=row["model_role_id"],
                partition="EVALUATION_HOLDOUT",
                validation_rows=int(row["rows"]),
                metrics={
                    "mean_residual": row["mean_residual"],
                    "p90_absolute_error": row["p90_absolute_error"],
                },
                status="completed",
                evidence_ref=f"holdout_predictions.csv#model_role_id={row['model_role_id']}",
            )
        )
    for name, frame, source in (
        ("Encoded estimator importance", encoded, "encoded_importance.csv"),
        ("Permutation MAE importance", permutation, "permutation_importance.csv"),
        ("Support-flagged subgroup metrics", subgroups, "subgroup_metrics.csv"),
    ):
        rows.append(
            _transcript_row(
                "explainability" if source != "subgroup_metrics.csv" else "subgroup",
                name,
                source,
                status="completed" if len(frame) else "unavailable",
                metrics={
                    "rows": len(frame),
                    "methods": sorted(frame["method"].dropna().astype(str).unique().tolist()) if "method" in frame else [],
                    "repeat_count": int(frame["repeat"].nunique()) if "repeat" in frame else None,
                    "small_sample_rows": int(frame["small_sample"].fillna(False).astype(bool).sum()) if "small_sample" in frame else None,
                },
            )
        )
    for item in uncertainty:
        rows.append(
            _transcript_row(
                "uncertainty",
                "Describe same-holdout q90 error band",
                "uncertainty.json",
                model_role_id=item.get("model_role_id"),
                status="completed",
                metrics=item,
                evidence_ref=f"uncertainty.json#model_role_id={item.get('model_role_id')}",
            )
        )
    for item in conclusions:
        rows.append(
            _transcript_row(
                "conclusion",
                "Record scoped model conclusion",
                "model_conclusions.json",
                model_role_id=item.get("model_role_id"),
                status=item.get("status"),
                finding=item.get("finding"),
                decision=item.get("decision"),
                limitation=item.get("limitation"),
                next_action=item.get("next_action"),
                evidence_ref="|".join(item.get("evidence_refs", [])),
            )
        )
    rows.append(
        _transcript_row(
            "operational",
            "Record non-activating operational assessment",
            "operational_assessment.json",
            status=operational.get("operational_status", "not_assessed"),
            decision="deployment_review_eligible" if operational.get("deployment_review_eligible") else "not_eligible",
            metrics=operational,
        )
    )
    rows.append(
        _transcript_row(
            "publication",
            "Publish complete immutable evidence pack",
            "conclusion.json",
            status=outcome.get("execution_status"),
            model_role_id=outcome.get("selected_model_id"),
            metrics={"scientific_outcome": outcome.get("scientific_outcome")},
            next_action=outcome.get("next_action"),
        )
    )
    transcript = pd.DataFrame(rows)
    transcript.insert(0, "sequence", range(1, len(transcript) + 1))
    return transcript


def load_training_validation_view(workspace: Path | str) -> dict[str, Any]:
    discovery = discover_latest_training_validation(workspace)
    if not discovery["available"]:
        return discovery
    root = Path(workspace).resolve(strict=True)
    manifest = discovery["manifest"]
    entries = _entry_map(manifest)
    partition = _read_json(root, entries, "partition_declaration.json")
    membership = _read_csv(root, entries, "partition_membership.csv")
    folds = _read_csv(root, entries, "fold_summary.csv")
    monthly = _read_csv(root, entries, "monthly_row_counts.csv")
    candidate_summary = _read_csv(root, entries, "candidate_summary.csv")
    candidate_folds = _read_csv(root, entries, "candidate_fold_metrics.csv")
    fit_diagnostics = _read_csv(root, entries, "fit_diagnostics.csv")
    runtime = _read_csv(root, entries, "runtime_summary.csv")
    performance = _read_csv(root, entries, "performance_comparison.csv")
    tradeoff = _read_csv(root, entries, "accuracy_runtime_tradeoff.csv")
    selection = _read_json(root, entries, "family_selection.json")
    conclusions = _read_json(root, entries, "model_conclusions.json")
    tuning_trials = _read_csv(root, entries, "tuning_trials.csv")
    tuning_summary = _read_json(root, entries, "tuning_summary.json")
    variants = _read_csv(root, entries, "variant_fold_metrics.csv")
    evaluation_declaration = _read_json(root, entries, "evaluation_declaration.json")
    holdout = _read_csv(root, entries, "holdout_metrics.csv")
    predictions = _read_csv(root, entries, "holdout_predictions.csv")
    encoded = _read_csv(root, entries, "encoded_importance.csv")
    permutation = _read_csv(root, entries, "permutation_importance.csv")
    subgroups = _read_csv(root, entries, "subgroup_metrics.csv")
    uncertainty = _read_json(root, entries, "uncertainty.json")
    operational = _read_json(root, entries, "operational_assessment.json")
    outcome = _read_json(root, entries, "conclusion.json")
    agent_summary = _read_json(root, entries, "agent_summary.json")
    ui_summary = _read_json(root, entries, "ui_summary.json")

    for filename, document in (
        ("partition_declaration.json", partition),
        ("evaluation_declaration.json", evaluation_declaration),
        ("agent_summary.json", agent_summary),
        ("ui_summary.json", ui_summary),
    ):
        if document.get("run_id") != discovery["run_id"]:
            raise ValueError(f"{filename} run identity does not match the manifest")
    _require_columns(
        membership,
        "partition_membership.csv",
        {"row_id", "source_position", "period", "partition"},
    )
    allowed_partitions = {"TRAIN", "EVALUATION_HOLDOUT", "INFERENCE_RESERVE"}
    if set(membership["partition"]) != allowed_partitions:
        raise ValueError("partition membership must contain only the three declared roles")
    actual_counts = membership["partition"].value_counts().to_dict()
    declared_counts = {key: int(value) for key, value in partition.get("counts", {}).items()}
    if actual_counts != declared_counts:
        raise ValueError("partition declaration counts do not reconcile with membership")
    if membership["row_id"].duplicated().any() or membership["source_position"].duplicated().any():
        raise ValueError("partition membership contains duplicate row identities")
    _require_columns(monthly, "monthly_row_counts.csv", {"period", "partition", "row_count"})
    monthly_counts = monthly.groupby("partition")["row_count"].sum().astype(int).to_dict()
    if monthly_counts != declared_counts:
        raise ValueError("monthly row counts do not reconcile with partition membership")

    _require_columns(folds, "fold_summary.csv", {"fold_id", "scope", "parent_fold_id", "ordinal", "parent_population_rows", "train_period_min", "train_period_max", "validation_period_min", "train_rows", "validation_rows", "not_used_yet_rows", "holdout_rows_excluded", "reserve_rows_excluded", "chronological_order_ok"})
    outer = folds[folds["scope"] == "outer"].copy()
    inner = folds[folds["scope"] == "inner"].copy()
    if len(outer) != 5 or len(inner) != 18:
        raise ValueError("fold_summary.csv must contain five outer and 18 inner folds")
    expected_outer = {f"outer-{index}" for index in range(1, 6)}
    if set(outer["fold_id"]) != expected_outer:
        raise ValueError("outer fold identities are incomplete")
    expected_parents = expected_outer | {"final-train"}
    inner_counts = inner.groupby("parent_fold_id").size().to_dict()
    if set(inner_counts) != expected_parents or set(inner_counts.values()) != {3}:
        raise ValueError("inner folds must contain three rows for each declared parent")
    for _, row in folds.iterrows():
        if int(row["train_rows"] + row["validation_rows"] + row["not_used_yet_rows"]) != int(row["parent_population_rows"]):
            raise ValueError("fold row counts do not reconcile with the parent population")
        if int(row["holdout_rows_excluded"]) != declared_counts["EVALUATION_HOLDOUT"] or int(row["reserve_rows_excluded"]) != declared_counts["INFERENCE_RESERVE"]:
            raise ValueError("fold exclusions do not reconcile with protected partitions")
    _require_columns(candidate_summary, "candidate_summary.csv", {"model", "configuration_id", "cv_mae_mean_usd", "cv_rmse_mean_usd", "cv_medae_mean_usd", "cv_r2_mean", "total_cv_fit_wall_s"})
    if set(candidate_summary["model"]) != CANDIDATES or len(candidate_summary) != 5:
        raise ValueError("candidate_summary.csv must contain exactly five candidate roles")
    _require_columns(candidate_folds, "candidate_fold_metrics.csv", {"model", "configuration_id", "fold_id", "fold_ordinal", "train_rows", "validation_rows", "train_MAE", "validation_MAE", "validation_RMSE", "validation_MedAE", "validation_R2", "fit_wall_s"})
    if len(candidate_folds) != 25 or set(candidate_folds["model"]) != CANDIDATES:
        raise ValueError("candidate fold evidence must contain five models by five folds")
    candidate_pairs = set(zip(candidate_folds["model"], candidate_folds["fold_id"], strict=True))
    if candidate_pairs != {(model, fold) for model in CANDIDATES for fold in expected_outer}:
        raise ValueError("candidate fold identities are incomplete or duplicated")
    _require_columns(variants, "variant_fold_metrics.csv", {"fold_id", "variant", "validation_MAE", "validation_RMSE", "validation_MedAE", "validation_R2"})
    if len(variants) != 10 or set(variants["variant"]) != {"full", "top2"}:
        raise ValueError("variant fold evidence must contain paired Full/Top-2 folds")
    variant_pairs = set(zip(variants["variant"], variants["fold_id"], strict=True))
    if variant_pairs != {(variant, fold) for variant in {"full", "top2"} for fold in expected_outer}:
        raise ValueError("variant fold identities are incomplete or duplicated")
    if set(tuning_summary) != expected_parents:
        raise ValueError("tuning summary must contain five outer and one final TRAIN context")
    _require_columns(holdout, "holdout_metrics.csv", {"model_role_id", "partition", "rows", "MAE", "RMSE", "MedAE", "R2"})
    if set(holdout["model_role_id"]) != ALL_ROLES or len(holdout) != 7:
        raise ValueError("holdout metrics must contain all seven model roles")
    if set(holdout["partition"]) != {"EVALUATION_HOLDOUT"}:
        raise ValueError("holdout metrics must contain EVALUATION_HOLDOUT only; reserve metrics are forbidden")
    if "partition" in predictions and set(predictions["partition"]) != {"EVALUATION_HOLDOUT"}:
        raise ValueError("holdout predictions must exclude inference reserve")
    if set(predictions["model_role_id"]) != ALL_ROLES:
        raise ValueError("holdout predictions must contain all seven model roles")
    numeric_metrics = candidate_folds[["train_MAE", "validation_MAE", "validation_RMSE", "validation_MedAE", "fit_wall_s"]].to_numpy(dtype=float)
    if not np.isfinite(numeric_metrics).all():
        raise ValueError("candidate metric evidence contains non-finite required values")
    if len(conclusions) != 7 or {item.get("model_role_id") for item in conclusions} != ALL_ROLES:
        raise ValueError("model conclusions must contain all seven model roles")

    prediction_summary = predictions.groupby("model_role_id", as_index=False).agg(
        rows=("row_id", "count"),
        mean_residual=("residual", "mean"),
        p90_absolute_error=(
            "absolute_error",
            lambda values: float(np.quantile(values, 0.9, method="linear")),
        ),
    )
    transcript = _build_transcript(
        partition=partition,
        monthly=monthly,
        folds=folds,
        candidate_folds=candidate_folds,
        selection=selection,
        tuning_trials=tuning_trials,
        tuning_summary=tuning_summary,
        variants=variants,
        evaluation_declaration=evaluation_declaration,
        holdout=holdout,
        prediction_summary=prediction_summary,
        encoded=encoded,
        permutation=permutation,
        subgroups=subgroups,
        uncertainty=uncertainty,
        conclusions=conclusions,
        operational=operational,
        outcome=outcome,
    )
    permutation_display = (
        permutation.groupby(["kind", "name", "scoring"], as_index=False)
        .agg(
            repeat_count=("repeat", "nunique"),
            mae_increase_mean_usd=("mae_increase", "mean"),
            mae_increase_sd_usd=("mae_increase", "std"),
            mae_increase_min_usd=("mae_increase", "min"),
            mae_increase_max_usd=("mae_increase", "max"),
        )
        if not permutation.empty
        else pd.DataFrame(
            columns=[
                "kind",
                "name",
                "scoring",
                "repeat_count",
                "mae_increase_mean_usd",
                "mae_increase_sd_usd",
                "mae_increase_min_usd",
                "mae_increase_max_usd",
            ]
        )
    )
    transcript_buffer = StringIO()
    transcript.to_csv(transcript_buffer, index=False)
    downloads = []
    for filename, mime in (
        ("training.log", "text/plain"),
        ("events.jsonl", "application/x-ndjson"),
        ("report.md", "text/markdown"),
        ("agent_summary.json", "application/json"),
        ("model_conclusions.json", "application/json"),
    ):
        downloads.append(
            {
                "label": f"Download {filename}",
                "filename": filename,
                "mime": mime,
                "content": _source_path(root, entries, filename).read_bytes(),
                "source_path": entries[filename]["path"],
                "sha256": entries[filename]["sha256"],
            }
        )
    downloads.append(
        {
            "label": "Download complete evidence-derived transcript (training-validation-transcript.csv)",
            "filename": "training-validation-transcript.csv",
            "mime": "text/csv",
            "content": transcript_buffer.getvalue().encode("utf-8"),
            "source_path": None,
            "sha256": None,
        }
    )
    candidate_conclusions = [item for item in conclusions if item["model_role_id"] in CANDIDATES]
    final_conclusions = [item for item in conclusions if item["model_role_id"] in FINAL_ROLES]
    fastest = candidate_summary.sort_values("total_cv_fit_wall_s").iloc[0]["model"]
    return {
        **discovery,
        "page04": {
            "partition": partition,
            "outer_folds": outer,
            "inner_folds": inner,
            "monthly_counts": monthly,
            "candidate_summary": candidate_summary,
            "candidate_folds": candidate_folds,
            "fit_diagnostics": fit_diagnostics,
            "runtime_summary": runtime,
            "performance": performance,
            "tradeoff": tradeoff,
            "selection": selection,
            "fastest_fit_model": fastest,
            "conclusions": candidate_conclusions,
        },
        "page05": {
            "tuning_trials": tuning_trials,
            "tuning_summary": tuning_summary,
            "variant_folds": variants,
            "holdout_metrics": holdout,
            "holdout_prediction_summary": prediction_summary,
            "encoded_importance": encoded,
            "permutation_importance": permutation_display,
            "subgroups": subgroups,
            "uncertainty": uncertainty,
            "operational": operational,
            "outcome": outcome,
            "conclusions": final_conclusions,
        },
        "transcript": transcript,
        "transcript_preview": transcript.head(MAX_PREVIEW_ROWS),
        "raw_log_preview": _source_path(root, entries, "training.log").read_text(encoding="utf-8")[:16_384],
        "event_preview": [json.loads(line) for line in _source_path(root, entries, "events.jsonl").read_text(encoding="utf-8").splitlines()[:200] if line.strip()],
        "report_preview": _source_path(root, entries, "report.md").read_text(encoding="utf-8")[:16_384],
        "downloads": downloads,
    }



def _display_frame(st, frame: pd.DataFrame, *, height: int = 320) -> None:
    display = frame.copy()
    for column in display.select_dtypes(include="object").columns:
        display[column] = display[column].map(
            lambda value: json.dumps(value, sort_keys=True)
            if isinstance(value, (dict, list, tuple))
            else value
        )
    st.dataframe(display, hide_index=True, width="stretch", height=height)


def sort_training_events(frame: pd.DataFrame) -> pd.DataFrame:
    """Sort audit events for reader review without changing their source values."""
    if frame.empty:
        return frame.copy().reset_index(drop=True)

    sorted_frame = frame.copy()
    for column in ("operation", "status", "model"):
        if column not in sorted_frame:
            sorted_frame[column] = None
    status_rank = {"started": 0, "completed": 1}
    sorted_frame["_operation_order"] = sorted_frame["operation"].fillna("").astype(str).str.casefold()
    sorted_frame["_status_order"] = (
        sorted_frame["status"].fillna("").astype(str).str.casefold().map(status_rank).fillna(2)
    )
    sorted_frame["_model_order"] = sorted_frame["model"].fillna("").astype(str).str.casefold()
    sort_columns = ["_operation_order", "_status_order", "_model_order"]
    if "sequence" in sorted_frame:
        sort_columns.append("sequence")
    return (
        sorted_frame.sort_values(sort_columns, kind="stable")
        .drop(columns=["_operation_order", "_status_order", "_model_order"])
        .reset_index(drop=True)
    )


def render_training_log_footer(st, audit: dict[str, Any], *, page: str) -> None:
    """Render the historical audit log as the final section of a model-review page."""
    st.divider()
    st.subheader("Training logs")
    with st.expander("Training log and audit downloads", expanded=False):
        if not audit.get("available"):
            st.info(f"Historical training audit unavailable: {audit.get('reason')}")
            return

        st.warning(
            "This audit records a historical pipeline run whose locked-test population was "
            "already scored. It is review evidence, not unseen future validation."
        )
        with st.container(horizontal=True):
            st.metric("Pipeline run", audit["pipeline_run_id"], border=True)
            st.metric("Audit status", audit["training_status"], border=True)
            st.metric("Audit events", audit["event_count"], border=True)
        st.caption(
            f"Audit `{audit['audit_run_id']}` matches the current source data and required "
            "audit checks. The table is limited to model-related events."
        )
        st.markdown(
            "Events are sorted by **Operation**, then **Status** (Started before Completed), "
            "then **Model**. Download the complete structured log for the original event order."
        )
        events = sort_training_events(pd.DataFrame(audit.get("event_preview", [])))
        _display_frame(st, events, height=480)
        if audit["relevant_event_count"] > len(events):
            st.caption(
                f"Preview shows {len(events)} of {audit['relevant_event_count']} "
                "model-related events."
            )
        for index, item in enumerate(audit["downloads"]):
            st.download_button(
                f"Download {item['label'].lower()}",
                item["content"],
                file_name=item["filename"],
                mime=item["mime"],
                key=f"{page}_historical_audit_{index}",
                width="stretch",
            )
