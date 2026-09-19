from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from ai_job_market.ui_evidence_io import EvidenceContractError

TABLE_EMPHASIS_LEGEND = "**Bold = governing result** · *Italic = reference/context*"


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _safe_audit_file(base: Path, relative_path: str) -> Path:
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("audit path is not safe")
    candidate = base / relative
    if candidate.is_symlink() or base.is_symlink():
        raise ValueError("audit path is not safe")
    resolved_base = base.resolve(strict=True)
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(resolved_base):
        raise ValueError("audit path is not safe")
    return resolved


def load_compatible_training_audit(root: Path) -> dict[str, Any]:
    """Load the newest complete source-compatible primary-pipeline audit trace."""
    workspace = root.resolve()
    raw_path = workspace / "data/raw/ai_jobs_market_2025_2026.csv"
    logs_root = workspace / "outputs/08_full_pipeline/logs"
    if not raw_path.is_file() or not logs_root.is_dir() or logs_root.is_symlink():
        return {
            "available": False,
            "reason": "Training audit source or log directory is unavailable.",
        }
    source_sha256 = _sha256_bytes(raw_path.read_bytes())
    candidates = sorted(logs_root.glob("training-*.logs"), reverse=True)
    if not candidates:
        return {"available": False, "reason": "No training audit runs are available."}

    last_reason = "No complete compatible training audit run was found."
    relevant_kinds = {"fold_membership", "tuning_trials", "model_comparison"}
    for log_path in candidates:
        try:
            if log_path.is_symlink() or not log_path.is_file():
                raise ValueError("audit path is not safe")
            run_dir = log_path.with_suffix("")
            manifest_path = _safe_audit_file(run_dir, "manifest.json")
            manifest_bytes = manifest_path.read_bytes()
            manifest = json.loads(manifest_bytes)
            complete = (
                manifest.get("training_status") == "completed"
                and manifest.get("log_complete") is True
                and manifest.get("console_complete") is True
                and manifest.get("coverage_complete") is True
            )
            if not complete:
                raise ValueError("training audit is not complete")
            if manifest.get("source_sha256") != source_sha256:
                raise ValueError("training audit source fingerprint does not match")

            log_bytes = log_path.read_bytes()
            events = [json.loads(line) for line in log_bytes.splitlines() if line.strip()]
            if not events:
                raise ValueError("training audit JSONL is empty")
            audit_run_id = str(manifest.get("audit_run_id", ""))
            pipeline_run_id = str(manifest.get("pipeline_run_id", ""))
            if any(
                str(event.get("audit_run_id", "")) != audit_run_id
                or (
                    event.get("pipeline_run_id") not in (None, "")
                    and str(event.get("pipeline_run_id")) != pipeline_run_id
                )
                for event in events
            ):
                raise ValueError("training audit JSONL run identity does not match manifest")

            downloads = [
                {
                    "label": "Audit manifest",
                    "kind": "audit_manifest",
                    "filename": f"training-{pipeline_run_id}-manifest.json",
                    "sha256": _sha256_bytes(manifest_bytes),
                    "content": manifest_bytes,
                    "mime": "application/json",
                },
                {
                    "label": "Complete structured training log",
                    "kind": "complete_jsonl",
                    "filename": log_path.name,
                    "sha256": _sha256_bytes(log_bytes),
                    "content": log_bytes,
                    "mime": "application/x-ndjson",
                },
            ]
            for entry in manifest.get("exports", []):
                if entry.get("status") != "complete" or entry.get("kind") not in relevant_kinds:
                    continue
                export_path = _safe_audit_file(run_dir, str(entry.get("path", "")))
                content = export_path.read_bytes()
                if _sha256_bytes(content) != entry.get("sha256"):
                    raise ValueError(f"training audit export checksum mismatch: {export_path.name}")
                downloads.append(
                    {
                        "label": str(entry.get("evidence_id") or export_path.stem),
                        "kind": str(entry["kind"]),
                        "filename": export_path.name,
                        "sha256": str(entry["sha256"]),
                        "content": content,
                        "mime": "text/csv",
                    }
                )
            selection = next(
                (
                    event
                    for event in reversed(events)
                    if event.get("operation") == "create_final_estimator"
                    and event.get("status") == "completed"
                ),
                {},
            )
            return {
                "available": True,
                "audit_run_id": audit_run_id,
                "pipeline_run_id": pipeline_run_id,
                "training_status": manifest["training_status"],
                "source_sha256": source_sha256,
                "selection_source": selection.get("selection_source"),
                "downloads": downloads,
            }
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            last_reason = str(exc)
    return {"available": False, "reason": last_reason}


def temporal_validation_guide(
    folds: pd.DataFrame, membership: pd.DataFrame, *, full_feature_count: int
) -> dict[str, Any]:
    """Summarize the implemented adjacent-block temporal evaluation from evidence."""
    required_fold = {
        "model",
        "fold_id",
        "validation_period",
        "train_rows",
        "validation_rows",
        "encoded_feature_count",
    }
    required_membership = {"fold_id", "partition_role", "record_id"}
    if folds.empty or not required_fold <= set(folds):
        raise ValueError("candidate fold evidence is incomplete")
    if membership.empty or not required_membership <= set(membership):
        raise ValueError("fold membership evidence is incomplete")
    definitions = (
        folds[
            [
                "fold_id",
                "validation_period",
                "train_rows",
                "validation_rows",
                "encoded_feature_count",
            ]
        ]
        .drop_duplicates(subset=["fold_id"])
        .sort_values("fold_id")
    )
    overlap_count = 0
    for fold_id in definitions.fold_id:
        current = membership[membership.fold_id == fold_id]
        train_ids = set(current.loc[current.partition_role == "train", "record_id"])
        validation_ids = set(current.loc[current.partition_role == "validation", "record_id"])
        overlap_count += len(train_ids & validation_ids)
    return {
        "fold_count": int(definitions.fold_id.nunique()),
        "candidate_count": int(folds.model.nunique()),
        "full_feature_count": int(full_feature_count),
        "row_overlap_count": int(overlap_count),
        "folds": definitions.to_dict("records"),
        "method": (
            "DEV rows are sorted chronologically and split into adjacent row blocks. "
            "Each fold fits preprocessing and one candidate model on one block, then scores "
            "the immediately following block."
        ),
        "limitations": (
            "This is not randomized K-fold and not an expanding-window design. Adjacent row "
            "blocks can share calendar-month labels even though their record identities are disjoint."
        ),
    }


def _parameter_values_match(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is None and right is None
    try:
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)
    except (TypeError, ValueError):
        return str(left) == str(right)


def tuning_method_guide(
    tables: dict[str, pd.DataFrame],
    *,
    applied_parameters: dict[str, Any],
    effective_folds: int,
) -> dict[str, Any]:
    """Summarize the inherited Random Forest tuning procedure from validated tables."""
    stage_columns = [
        ("n_estimators", "manual_tuning_step2_n_estimators"),
        ("max_depth", "manual_tuning_step3_max_depth"),
        ("min_samples_leaf", "manual_tuning_step4_min_samples_leaf"),
        ("max_features", "manual_tuning_step5_max_features"),
    ]
    initial = tables.get("manual_tuning_step1_gridsearch", pd.DataFrame())
    if initial.empty or "CV_R2" not in initial:
        raise ValueError("initial tuning grid evidence is incomplete")
    stage_counts: dict[str, int] = {}
    for parameter, stem in stage_columns:
        frame = tables.get(stem, pd.DataFrame())
        if frame.empty or parameter not in frame or "CV_R2" not in frame:
            raise ValueError(f"tuning stage evidence is incomplete: {parameter}")
        stage_counts[parameter] = int(len(frame))
    winner = initial.sort_values("CV_R2", ascending=False, kind="stable").iloc[0]
    compared = ["n_estimators", "max_depth", "min_samples_leaf", "max_features"]
    saved_matches = all(
        _parameter_values_match(winner.get(parameter), applied_parameters.get(parameter))
        for parameter in compared
    )
    return {
        "initial_trial_count": int(len(initial)),
        "stage_trial_counts": stage_counts,
        "total_trial_count": int(len(initial) + sum(stage_counts.values())),
        "effective_folds": int(effective_folds),
        "ranking_metric": "CV R²",
        "ranking_direction": "higher is better",
        "saved_matches_initial_winner": saved_matches,
        "applied_parameters": {name: applied_parameters.get(name) for name in compared},
        "selection_scope": "Inherited non-nested DEV temporal validation; locked test not used.",
    }


def load_evidence_download(root: Path, manifest: dict[str, Any], stem: str) -> dict[str, Any]:
    """Return byte-exact content for one validated active evidence file."""
    matches = [entry for entry in manifest["files"] if Path(entry["path"]).stem == stem]
    if len(matches) != 1:
        raise EvidenceContractError(f"evidence file is unavailable or ambiguous: {stem}")
    entry = matches[0]
    workspace = root.resolve()
    path = root / entry["path"]
    if path.is_symlink():
        raise EvidenceContractError(f"evidence download path is unsafe: {entry['path']}")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise EvidenceContractError(f"evidence download is unavailable: {entry['path']}") from exc
    if not resolved.is_relative_to(workspace):
        raise EvidenceContractError(f"evidence download path escapes workspace: {entry['path']}")
    content = resolved.read_bytes()
    actual_sha256 = _sha256_bytes(content)
    if actual_sha256 != entry["sha256"]:
        raise EvidenceContractError(f"evidence download checksum mismatch: {entry['path']}")
    return {
        "path": entry["path"],
        "filename": resolved.name,
        "sha256": actual_sha256,
        "content": content,
        "mime": "text/csv" if resolved.suffix == ".csv" else "application/octet-stream",
    }


def _markdown_emphasis(value: str, emphasis: str | None) -> str:
    if emphasis == "bold":
        return f"**{value}**"
    if emphasis == "italic":
        return f"*{value}*"
    return value


def _display_parameter(value: Any) -> str:
    if value is None or pd.isna(value):
        return "None"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def candidate_decision_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Build the small Markdown-enabled Page 04 decision table."""
    required = {
        "model",
        "validation_MAE_mean",
        "train_MAE_mean",
        "validation_R2_mean",
    }
    if summary.empty or not required <= set(summary):
        return pd.DataFrame(
            columns=[
                "Rank",
                "Model",
                "Decision role",
                "Mean validation MAE",
                "Mean train MAE",
                "Validation R²",
            ]
        )
    table = summary.copy()
    table["validation_MAE_mean"] = pd.to_numeric(table["validation_MAE_mean"], errors="coerce")
    table = table.dropna(subset=["validation_MAE_mean"]).sort_values(
        ["validation_MAE_mean", "model"], kind="stable"
    )
    table["Rank"] = table["validation_MAE_mean"].rank(method="min").astype(int)
    best = float(table["validation_MAE_mean"].min())
    rows: list[dict[str, Any]] = []
    for _, row in table.iterrows():
        is_winner = math.isclose(float(row.validation_MAE_mean), best, rel_tol=0.0, abs_tol=0.0)
        is_reference = row.model == "Dummy Median"
        emphasis = "bold" if is_winner else "italic" if is_reference else None
        role = "Winner" if is_winner else "Reference baseline" if is_reference else "Candidate"
        rows.append(
            {
                "Rank": int(row.Rank),
                "Model": _markdown_emphasis(str(row.model), emphasis),
                "Decision role": _markdown_emphasis(role, emphasis),
                "Mean validation MAE": _markdown_emphasis(
                    f"${float(row.validation_MAE_mean):,.0f}", emphasis
                ),
                "Mean train MAE": _markdown_emphasis(
                    f"${float(row.train_MAE_mean):,.0f}", emphasis
                ),
                "Validation R²": _markdown_emphasis(
                    f"{float(row.validation_R2_mean):.3f}", emphasis
                ),
            }
        )
    return pd.DataFrame(rows)


def tuning_decision_table(
    summary: pd.DataFrame, applied_parameters: dict[str, Any]
) -> pd.DataFrame:
    """Build the small Markdown-enabled Page 05 tuning decision table."""
    required = {
        "hyperparameter",
        "search_space",
        "optimal_value",
        "best_cv_r2",
        "best_cv_mae",
    }
    if summary.empty or not required <= set(summary):
        return pd.DataFrame(
            columns=[
                "Hyperparameter",
                "Search space",
                "Stage winner",
                "Applied value",
                "Relationship",
                "Best CV R²",
                "Best CV MAE",
            ]
        )
    rows: list[dict[str, Any]] = []
    for _, row in summary.iterrows():
        parameter = str(row.hyperparameter).split(". ", 1)[-1]
        winner = row.optimal_value
        applied = applied_parameters.get(parameter)
        matches = _parameter_values_match(winner, applied)
        rows.append(
            {
                "Hyperparameter": parameter,
                "Search space": f"*{row.search_space}*",
                "Stage winner": _markdown_emphasis(_display_parameter(winner), "bold"),
                "Applied value": _markdown_emphasis(_display_parameter(applied), "bold"),
                "Relationship": (
                    "**Applied = stage winner**" if matches else "*Sensitivity only; not applied*"
                ),
                "Best CV R²": _markdown_emphasis(f"{float(row.best_cv_r2):.3f}", "bold"),
                "Best CV MAE": _markdown_emphasis(f"${float(row.best_cv_mae):,.0f}", "bold"),
            }
        )
    return pd.DataFrame(rows)
