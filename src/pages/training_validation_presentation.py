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
            "No complete compatible training-validation pack is available.",
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
                    metrics={"mae_mean": row.get("mae_mean"), "mae_sd": row.get("mae_sd")},
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


def _audit_identity(manifest: dict[str, Any], audit: dict[str, Any]) -> str:
    return (
        f"pipeline {audit.get('pipeline_run_id', manifest.get('source_run_id', 'unavailable'))}; "
        f"audit {audit.get('audit_run_id', 'unavailable')}; evidence "
        f"{manifest.get('evidence_id', 'unavailable')}"
    )


def build_historical_candidate_5w1h(
    manifest: dict[str, Any], tables: dict[str, pd.DataFrame], audit: dict[str, Any]
) -> list[dict[str, Any]]:
    """Build five evidence-bound historical candidate narratives."""
    summary = tables["candidate_summary"].sort_values(
        "validation_MAE_mean", kind="stable"
    )
    folds = tables["candidate_fold_metrics"]
    if set(summary["model"]) != CANDIDATES or len(summary) != 5:
        raise ValueError("historical candidate evidence must contain five model roles")
    identity = _audit_identity(manifest, audit)
    features = list(manifest.get("full_features", []))
    winner = str(summary.iloc[0]["model"])
    records: list[dict[str, Any]] = []
    for _, row in summary.iterrows():
        model = str(row["model"])
        model_folds = folds[folds["model"] == model].sort_values("fold_id")
        if len(model_folds) != int(row["effective_folds"]):
            raise ValueError(f"historical fold evidence does not reconcile for {model}")
        periods = model_folds["validation_period"].astype(str).tolist()
        if model == "Dummy Median":
            decision = "Reference floor used to test whether learned candidates add predictive value."
        elif model == winner:
            decision = "Lowest mean validation MAE among the five frozen candidate configurations."
        else:
            decision = "Comparison candidate retained to show relative error, stability and runtime."
        records.append(
            {
                "model_role_id": model,
                "configuration_id": str(row["configuration_id"]),
                "who": f"{identity}; candidate role {model}.",
                "what": (
                    f"Continuous annual_salary_usd regression using {len(features)} approved "
                    "raw feature families."
                ),
                "when": f"Five historical DEV validation periods: {'; '.join(periods)}.",
                "where": "Historical DEV temporal-validation evidence; not strict future-reserve validation.",
                "why": decision,
                "how": (
                    "Fold-local preprocessing and estimator fitting on the training block, followed "
                    "by scoring on the adjacent chronological validation block; configuration "
                    f"{row['configuration_id']}."
                ),
                "result": {
                    "folds": int(row["effective_folds"]),
                    "train_mae_usd": float(row["train_MAE_mean"]),
                    "validation_mae_usd": float(row["validation_MAE_mean"]),
                    "validation_mae_sd_usd": float(row["validation_MAE_SD"]),
                    "validation_rmse_usd": float(row["validation_RMSE_mean"]),
                    "validation_medae_usd": float(row["validation_MedAE_mean"]),
                    "validation_r2": float(row["validation_R2_mean"]),
                    "mean_fit_s": float(row["fit_time_mean_s"]),
                },
                "limitation": (
                    "This historical DEV comparison does not establish untouched future "
                    "performance, causality, fairness or deployment readiness."
                ),
                "next_action": "Review fold rows and raw audit events before accepting the comparison.",
                "evidence_refs": [
                    f"candidate_summary.csv#model={model}",
                    f"candidate_fold_metrics.csv#model={model}",
                ],
            }
        )
    return records


def build_historical_final_5w1h(
    manifest: dict[str, Any], tables: dict[str, pd.DataFrame], audit: dict[str, Any]
) -> list[dict[str, Any]]:
    """Build selected-family and Full/Top-2 historical records."""
    candidates = build_historical_candidate_5w1h(manifest, tables, audit)
    selected = candidates[0]
    records = [
        {
            **selected,
            "model_role_id": f"{selected['model_role_id']} (selected family)",
            "why": (
                "Selected by the historical DEV candidate ranking; locked-test results do not "
                "retroactively change that family decision."
            ),
            "limitation": (
                "The later locked-test population is historically exposed; this selected-family "
                "record is not strict future-reserve validation."
            ),
        }
    ]
    variants = tables["variant_metrics"]
    identity = _audit_identity(manifest, audit)
    for role, model_name, model_key in (
        ("full", "Full 13", "full:selected"),
        ("top2", "Top 2", "top2:fixed"),
    ):
        dev = variants[(variants["feature_variant"] == role) & (variants["evaluation"] == "dev_cv_mean")]
        historical = variants[(variants["feature_variant"] == role) & (variants["evaluation"] == "historical_test")]
        if len(dev) != 1 or len(historical) != 1:
            raise ValueError(f"historical variant evidence does not reconcile for {role}")
        dev_row = dev.iloc[0]
        test_row = historical.iloc[0]
        model = manifest.get("models", {}).get(model_key, {})
        feature_order = list(model.get("feature_order", []))
        parameters = dict(model.get("parameters", {}))
        records.append(
            {
                "model_role_id": model_name,
                "configuration_id": str(dev_row["configuration_id"]),
                "who": f"{identity}; frozen {model_name} Random Forest role.",
                "what": (
                    "Fixed Top-2 feature variant (job_category and years_of_experience)."
                    if role == "top2"
                    else f"Full estimator using {len(feature_order)} approved raw features."
                ),
                "when": (
                    "Historical five-fold DEV evidence followed by the previously scored locked-test population."
                ),
                "where": "DEV CV plus historically exposed locked-test evidence; inference reserve is not scored.",
                "why": (
                    "Controlled fixed-feature comparison only; no automatic promotion."
                    if role == "top2"
                    else "Saved full-feature reference retained for granular model-reliance evidence."
                ),
                "how": (
                    "Frozen Random Forest pipeline with inherited applied settings "
                    f"n_estimators={parameters.get('n_estimators')}, "
                    f"max_depth={parameters.get('max_depth')}, "
                    f"min_samples_leaf={parameters.get('min_samples_leaf')}, and "
                    f"max_features={parameters.get('max_features')}."
                ),
                "result": {
                    "dev_cv_mae_usd": float(dev_row["MAE"]),
                    "dev_cv_r2": float(dev_row["R2"]),
                    "historical_test_rows": int(test_row["row_count"]),
                    "historical_test_mae_usd": float(test_row["MAE"]),
                    "historical_test_rmse_usd": float(test_row["RMSE"]),
                    "historical_test_medae_usd": float(test_row["MedAE"]),
                    "historical_test_r2": float(test_row["R2"]),
                    "q90_abs_error_usd": float(test_row["q90_abs_error_usd"]),
                    "same_population_coverage": float(test_row["coverage"]),
                },
                "limitation": (
                    "Locked-test rows are historically exposed; q90 coverage is descriptive on "
                    "the same population and the variant is not automatically promoted."
                ),
                "next_action": "Use for transparent historical review, not unseen-validation claims.",
                "evidence_refs": [
                    f"variant_metrics.csv#feature_variant={role}&evaluation=dev_cv_mean",
                    f"variant_metrics.csv#feature_variant={role}&evaluation=historical_test",
                    f"manifest.json#models.{model_key}",
                ],
            }
        )
    return records


def _render_unavailable(st, view: dict[str, Any], *, key: str) -> None:
    st.info("Training validation unavailable")
    st.caption(
        "Reason: "
        + view.get("message", "No complete compatible offline evidence pack exists.")
    )
    st.caption("No fallback evidence is displayed.")
    st.code(view["inspect_command"], language="bash")
    st.markdown(f"**Next action:** {view['next_action']}")
    if view.get("invalid_candidate_count"):
        st.caption(
            f"Rejected final-pack candidates: {view['invalid_candidate_count']} "
            "(no fixture, historical, supplemental, or hardcoded fallback was used)."
        )


def _render_identity(st, view: dict[str, Any]) -> None:
    st.success("Validated immutable offline evidence pack")
    st.caption(
        f"Run `{view['run_id']}` · generated `{view['generated_at']}` · "
        f"manifest `{view['manifest_sha256'][:12]}…` · schema `training-validation/v1`"
    )
    if view["invalid_candidate_count"]:
        st.caption(
            f"Selected latest valid pack after rejecting "
            f"{view['invalid_candidate_count']} incompatible candidate(s)."
        )


def _render_downloads(st, view: dict[str, Any], names: set[str], *, prefix: str) -> None:
    for index, item in enumerate(view["downloads"]):
        if item["filename"] not in names:
            continue
        st.download_button(
            item["label"],
            item["content"],
            file_name=item["filename"],
            mime=item["mime"],
            key=f"{prefix}_{index}",
            width="stretch",
        )


def _display_frame(st, frame: pd.DataFrame, *, height: int = 320) -> None:
    display = frame.copy()
    for column in display.select_dtypes(include="object").columns:
        display[column] = display[column].map(
            lambda value: json.dumps(value, sort_keys=True)
            if isinstance(value, (dict, list, tuple))
            else value
        )
    st.dataframe(display, hide_index=True, width="stretch", height=height)


def _load_view_for_ui(workspace: Path | str) -> dict[str, Any]:
    try:
        return load_training_validation_view(workspace)
    except (EvidenceContractError, OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        return _unavailable(
            "PACK_PROJECTION_FAILED",
            "The pack passed discovery but its UI projection is inconsistent. "
            f"Review the offline evidence contract ({type(exc).__name__}).",
        )


def _render_visible_overview(st, view: dict[str, Any], *, page: str) -> None:
    """Show trust context without changing the existing analytical sections."""
    with st.container(border=True):
        st.subheader("Training & validation evidence")
        if not view["available"]:
            st.warning("No verified training-validation evidence for this workspace")
            st.markdown(
                "The supported offline workflow covers chronological partitioning, expanding "
                "validation, candidate comparison, bounded tuning, one-time holdout evaluation, "
                "explainability, uncertainty, and immutable evidence publication. This describes "
                "the review capability—not a completed or trusted run for the active workspace."
            )
            st.caption(
                "Open the detailed section below for the safe reason and read-only inspection "
                "command. No fallback results are shown."
            )
            return
        st.success("Verified offline training evidence")
        if page == "page04":
            evidence = view["page04"]
            metrics = (
                ("Outer folds", len(evidence["outer_folds"])),
                ("Inner folds", len(evidence["inner_folds"])),
                ("Candidates", len(evidence["candidate_summary"])),
                ("Candidate-fold results", len(evidence["candidate_folds"])),
            )
        else:
            evidence = view["page05"]
            metrics = (
                ("Tuning contexts", len(evidence["tuning_summary"])),
                ("Variant folds", len(evidence["variant_folds"])),
                ("Holdout roles", len(evidence["holdout_metrics"])),
                ("Final conclusions", len(evidence["conclusions"])),
            )
        with st.container(horizontal=True):
            for label, value in metrics:
                st.metric(label, value, border=True)
        st.caption(
            f"Validated run `{view['run_id']}` · generated `{view['generated_at']}` · "
            "read-only evidence display, not live training or model activation."
        )
        st.markdown(
            "**Trust boundary:** tuning uses TRAIN-only temporal evidence; final metrics use "
            "`EVALUATION_HOLDOUT`; `INFERENCE_RESERVE` has no metrics or predictions."
        )


def _historical_report_markdown(
    records: list[dict[str, Any]], *, page: str, manifest: dict[str, Any], audit: dict[str, Any]
) -> str:
    lines = [
        "# Historical Branch B Training Report",
        "",
        f"Page scope: {page}",
        f"Supplemental evidence: {manifest.get('evidence_id', 'unavailable')}",
        f"Pipeline run: {audit.get('pipeline_run_id', 'unavailable')}",
        f"Audit run: {audit.get('audit_run_id', 'unavailable')}",
        "",
        "> Historical primary-pipeline and supplemental evidence. Locked-test rows are already "
        "exposed; this is not strict future-reserve validation.",
        "",
    ]
    for record in records:
        lines.extend(
            [
                f"## {record['model_role_id']}",
                "",
                f"- **Who:** {record['who']}",
                f"- **What:** {record['what']}",
                f"- **When:** {record['when']}",
                f"- **Where:** {record['where']}",
                f"- **Why:** {record['why']}",
                f"- **How:** {record['how']}",
                f"- **Result:** `{json.dumps(record['result'], sort_keys=True)}`",
                f"- **Limitation:** {record['limitation']}",
                f"- **Next action:** {record['next_action']}",
                f"- **Sources:** {' | '.join(record['evidence_refs'])}",
                "",
            ]
        )
    return "\n".join(lines)


def render_historical_training_report(
    st,
    manifest: dict[str, Any],
    tables: dict[str, pd.DataFrame],
    audit: dict[str, Any],
    *,
    page: str,
) -> None:
    """Render an additive historical report without changing existing analytics."""
    if not audit.get("available"):
        with st.container(border=True):
            st.subheader("Historical Branch B training report")
            st.warning(f"Historical training audit unavailable: {audit.get('reason')}")
        return
    records = (
        build_historical_candidate_5w1h(manifest, tables, audit)
        if page == "page04"
        else build_historical_final_5w1h(manifest, tables, audit)
    )
    with st.container(border=True):
        st.subheader("Historical Branch B training report")
        st.warning(
            "Historical primary-pipeline + supplemental evidence. The locked-test population was "
            "already scored; this report is not strict unseen training-validation evidence."
        )
        with st.container(horizontal=True):
            st.metric("Pipeline run", audit["pipeline_run_id"], border=True)
            st.metric("Audit status", audit["training_status"], border=True)
            st.metric("5W1H records", len(records), border=True)
            st.metric("Raw audit events", audit["event_count"], border=True)
        st.markdown(
            "**Evidence flow:** common prepared feature base → Branch A analytical workflow → "
            "Branch B salary comparison, tuning and historical evaluation. Branch A cluster "
            "assignments are not salary-model features."
        )
        st.caption(
            f"Supplemental evidence `{manifest['evidence_id']}` · audit "
            f"`{audit['audit_run_id']}` · complete source-compatible log."
        )

    with st.expander(
        "Model-by-model 5W1H evidence" if page == "page04" else "Selected and final model 5W1H evidence",
        expanded=False,
    ):
        for record in records:
            with st.container(border=True):
                st.markdown(f"#### {record['model_role_id']}")
                st.caption(f"Configuration `{record['configuration_id']}`")
                for question in ("who", "what", "when", "where", "why", "how"):
                    st.markdown(f"**{question.title()}:** {record[question]}")
                result = pd.DataFrame(
                    {
                        "Metric": list(record["result"]),
                        "Observed value": list(record["result"].values()),
                    }
                )
                _display_frame(st, result, height=min(320, 42 + 35 * len(result)))
                st.markdown(f"**Limitation:** {record['limitation']}")
                st.markdown(f"**Next action:** {record['next_action']}")
                st.caption("Sources: " + " · ".join(record["evidence_refs"]))

    with st.expander("Historical training activity log and downloads", expanded=False):
        st.markdown(
            "The table is a bounded projection of raw model-relevant JSONL events. It is not the "
            "strict training-validation transcript. Download the complete original log for audit."
        )
        _display_frame(st, pd.DataFrame(audit["event_preview"]), height=480)
        if audit["relevant_event_count"] > len(audit["event_preview"]):
            st.caption(
                f"Preview shows {len(audit['event_preview'])} of "
                f"{audit['relevant_event_count']} model-relevant events."
            )
        report = _historical_report_markdown(
            records, page=page, manifest=manifest, audit=audit
        ).encode("utf-8")
        st.download_button(
            "Download historical 5W1H report",
            report,
            file_name=f"historical-branch-b-{page}-5w1h-report.md",
            mime="text/markdown",
            key=f"{page}_historical_report",
            width="stretch",
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


def render_page04_training_validation(st, workspace: Path | str) -> None:
    """Render partition, fold, candidate, and log evidence on Page 04."""
    view = _load_view_for_ui(workspace)
    _render_visible_overview(st, view, page="page04")
    with st.expander(
        "Latest training validation: folds and candidate models",
        expanded=False,
    ):
        if not view["available"]:
            _render_unavailable(st, view, key="p4_tv")
            return
        _render_identity(st, view)
        evidence = view["page04"]
        partition = evidence["partition"]
        counts = partition.get("counts", {})
        st.markdown("### Approved chronological partition")
        st.markdown(
            f"**TRAIN:** {counts.get('TRAIN', '—')} rows · "
            f"**EVALUATION_HOLDOUT:** {counts.get('EVALUATION_HOLDOUT', '—')} rows · "
            f"**INFERENCE_RESERVE:** {counts.get('INFERENCE_RESERVE', '—')} rows. "
            "Whole-month chronology takes precedence over exact percentages. "
            "The reserve is inference-only and was not scored."
        )
        _display_frame(st, evidence["monthly_counts"])

        st.markdown("### Five expanding monthly outer folds")
        st.caption(
            "Every denominator below is the 80-row TRAIN partition; evaluation holdout and "
            "inference reserve rows remain excluded."
        )
        for row in evidence["outer_folds"].sort_values("ordinal").itertuples():
            added = "initial history" if pd.isna(row.added_train_rows) else f"+{int(row.added_train_rows)} rows"
            st.markdown(
                f"**{row.fold_id}** — parent TRAIN `{int(row.parent_population_rows)}` rows; "
                f"train `{row.train_period_min}`–`{row.train_period_max}` "
                f"({int(row.train_rows)} rows); validate `{row.validation_period_min}` "
                f"({int(row.validation_rows)} rows); added history `{added}`; "
                f"not used yet `{int(row.not_used_yet_rows)}`; holdout excluded "
                f"`{int(row.holdout_rows_excluded)}`; reserve excluded "
                f"`{int(row.reserve_rows_excluded)}`; overlap `{int(row.row_overlap_count)}`; "
                f"chronology `{'pass' if row.chronological_order_ok else 'fail'}`."
            )
        _display_frame(st, evidence["outer_folds"])

        st.markdown("### Eighteen parent-labelled inner folds")
        st.caption(
            "Three expanding monthly inner folds belong to each of five outer-training "
            "contexts plus the final TRAIN context. These are tuning evidence, not holdout evidence."
        )
        parent_counts = (
            evidence["inner_folds"].groupby(["parent_fold_id", "search_id"], dropna=False)
            .size()
            .rename("inner_fold_count")
            .reset_index()
        )
        _display_frame(st, parent_counts, height=250)
        _display_frame(st, evidence["inner_folds"], height=420)

        st.markdown("### Five candidates × five outer folds")
        _display_frame(st, evidence["candidate_summary"])
        _display_frame(st, evidence["candidate_folds"], height=500)
        st.caption(
            f"Lowest-MAE family: `{evidence['selection'].get('lowest_mae_model')}` · "
            f"selected under the declared overlap/simplicity rule: "
            f"`{evidence['selection'].get('selected_family')}` · fastest candidate fit: "
            f"`{evidence['fastest_fit_model']}`. Runtime does not override scientific selection."
        )
        _display_frame(st, evidence["fit_diagnostics"], height=360)
        _display_frame(st, evidence["tradeoff"])

        st.markdown("### Per-candidate conclusions")
        for item in evidence["conclusions"]:
            st.markdown(
                f"**{item['model_role_id']} — {item.get('decision', 'unavailable')}**  \n"
                f"Finding: {item.get('finding')}  \n"
                f"Limitation: {item.get('limitation')}  \n"
                f"Next action: {item.get('next_action')}  \n"
                f"Source: `{' | '.join(item.get('evidence_refs', []))}`"
            )

        st.markdown("### Logs and evidence-derived transcript")
        st.warning(
            "`training.log` is a short human log and `events.jsonl` is the raw structured "
            "lifecycle stream. The table below is a deterministic evidence-derived transcript; "
            "its sequence expresses method order and does not invent execution timestamps."
        )
        raw_tab, event_tab, transcript_tab = st.tabs(
            ["Raw human log", "Raw lifecycle events", "Evidence-derived transcript"]
        )
        with raw_tab:
            st.code(view["raw_log_preview"], language="text")
        with event_tab:
            _display_frame(st, pd.DataFrame(view["event_preview"]))
        with transcript_tab:
            _display_frame(st, view["transcript_preview"], height=500)
            if len(view["transcript"]) > len(view["transcript_preview"]):
                st.caption(
                    f"Preview shows {len(view['transcript_preview'])} of "
                    f"{len(view['transcript'])} rows; the CSV download is complete."
                )
        _render_downloads(
            st,
            view,
            {
                "training.log",
                "events.jsonl",
                "report.md",
                "agent_summary.json",
                "model_conclusions.json",
                "training-validation-transcript.csv",
            },
            prefix="p4_tv_download",
        )


def render_page05_training_validation(st, workspace: Path | str) -> None:
    """Render visible trust context plus two collapsed Page 05 evidence expanders."""
    view = _load_view_for_ui(workspace)
    _render_visible_overview(st, view, page="page05")

    with st.expander(
        "Latest training validation: tuning and Full/Top-2 variants",
        expanded=False,
    ):
        if not view["available"]:
            _render_unavailable(st, view, key="p5_tv_tuning")
        else:
            _render_identity(st, view)
            evidence = view["page05"]
            st.markdown("### Six nested tuning contexts")
            st.caption(
                "The selected family is tuned only inside each outer-training context and once "
                "inside final TRAIN. Reused and skipped slots remain visible with their reasons."
            )
            tuning_contexts = pd.DataFrame(
                [
                    {
                        "Context": context,
                        "Status": summary.get("status"),
                        "Reason": summary.get("reason"),
                        "Actual fits": summary.get("actual_fit_count"),
                        "Winner": json.dumps(summary.get("winner"), sort_keys=True),
                    }
                    for context, summary in sorted(evidence["tuning_summary"].items())
                ]
            )
            _display_frame(st, tuning_contexts)
            _display_frame(st, evidence["tuning_trials"], height=500)
            st.markdown("### Matched Full versus Top-2 outer-fold evaluation")
            st.caption(
                "Ten rows represent two feature variants on the same five outer folds. Top-2 uses "
                "`job_category` and `years_of_experience`; this comparison occurs before holdout access."
            )
            _display_frame(st, evidence["variant_folds"], height=420)
            tuning_stages = {"inner_fold", "tuning_trial", "variant_fold", "final_fit"}
            _display_frame(
                st,
                view["transcript"][view["transcript"]["stage"].isin(tuning_stages)].head(MAX_PREVIEW_ROWS),
                height=420,
            )

    with st.expander(
        "Latest training validation: holdout, explainability and uncertainty",
        expanded=False,
    ):
        if not view["available"]:
            _render_unavailable(st, view, key="p5_tv_final")
        else:
            _render_identity(st, view)
            evidence = view["page05"]
            st.markdown("### Final TRAIN fits and one-time evaluation holdout")
            st.caption(
                "All five frozen candidates and the two final variants are scored on "
                "EVALUATION_HOLDOUT only. INFERENCE_RESERVE has no metrics or predictions. "
                "Holdout ranking does not alter the frozen selection."
            )
            _display_frame(st, evidence["holdout_metrics"])
            _display_frame(st, evidence["holdout_prediction_summary"])
            st.markdown("### Explainability, subgroup, and uncertainty evidence")
            st.caption(
                "Encoded importance describes the fitted estimator; permutation values are MAE "
                "increases. Subgroups retain support/small-sample flags. The q90 band is descriptive "
                "same-holdout evidence, not a calibrated deployment interval."
            )
            _display_frame(st, evidence["encoded_importance"])
            _display_frame(st, evidence["permutation_importance"], height=360)
            _display_frame(st, evidence["subgroups"])
            _display_frame(st, pd.DataFrame(evidence["uncertainty"]))
            st.markdown("### Scientific and operational outcome")
            scientific = evidence["outcome"].get("scientific_outcome", {})
            st.markdown(
                f"Scientific status: **{scientific.get('status', 'unavailable')}** · "
                f"selected role/family: `{evidence['outcome'].get('selected_model_id', 'unavailable')}`."
            )
            st.markdown(
                f"Operational status: **{evidence['operational'].get('operational_status', 'unavailable')}** · "
                f"deployment-review eligible: "
                f"`{evidence['operational'].get('deployment_review_eligible', False)}`. "
                "Scientific acceptability does not activate or deploy a model."
            )
            for item in evidence["conclusions"]:
                st.markdown(
                    f"**{item['model_role_id']} — {item.get('decision', 'unavailable')}**  \n"
                    f"Finding: {item.get('finding')}  \n"
                    f"Limitation: {item.get('limitation')}  \n"
                    f"Next action: {item.get('next_action')}  \n"
                    f"Source: `{' | '.join(item.get('evidence_refs', []))}`"
                )
            st.markdown("### Report, raw logs, summaries, and complete transcript")
            st.warning(
                "Raw log/event/report files are original byte-identical pack files. The transcript "
                "CSV is generated in memory from authoritative evidence and is explicitly derived."
            )
            st.markdown(view["report_preview"])
            final_stages = {
                "final_fit", "holdout", "holdout_residual", "explainability", "subgroup", "uncertainty",
                "conclusion", "operational", "publication",
            }
            _display_frame(
                st,
                view["transcript"][view["transcript"]["stage"].isin(final_stages)].head(MAX_PREVIEW_ROWS),
                height=420,
            )
            _render_downloads(
                st,
                view,
                {
                    "training.log",
                    "events.jsonl",
                    "report.md",
                    "agent_summary.json",
                    "model_conclusions.json",
                    "training-validation-transcript.csv",
                },
                prefix="p5_tv_download",
            )
