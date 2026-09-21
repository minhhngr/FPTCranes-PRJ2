from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from ai_job_market.ui_evidence_io import EvidenceContractError
from components.presentation import _src, _tr

TABLE_EMPHASIS_LEGEND = _src(
    "model_training_presentation.bold_governing_result_italic_reference_context_ea97212"
)


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _safe_audit_file(base: Path, relative_path: str) -> Path:
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(_src("model_training_presentation.audit_path_is_not_safe_25def6f"))
    candidate = base / relative
    if candidate.is_symlink() or base.is_symlink():
        raise ValueError(_src("model_training_presentation.audit_path_is_not_safe_25def6f"))
    resolved_base = base.resolve(strict=True)
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(resolved_base):
        raise ValueError(_src("model_training_presentation.audit_path_is_not_safe_25def6f"))
    return resolved


def load_compatible_training_audit(root: Path) -> dict[str, Any]:
    """Load the newest complete source-compatible primary-pipeline audit trace."""
    workspace = root.resolve()
    raw_path = workspace / "data/raw/ai_jobs_market_2025_2026.csv"
    logs_root = workspace / "outputs/08_full_pipeline/logs"
    if not raw_path.is_file() or not logs_root.is_dir() or logs_root.is_symlink():
        return {
            "available": False,
            "reason": _src(
                "model_training_presentation.training_audit_source_or_log_directory_is_bf4035b"
            ),
        }
    source_sha256 = _sha256_bytes(raw_path.read_bytes())
    candidates = sorted(logs_root.glob("training-*.logs"), reverse=True)
    if not candidates:
        return {
            "available": False,
            "reason": _src(
                "model_training_presentation.no_training_audit_runs_are_available_4b11161"
            ),
        }

    last_reason = _src(
        "model_training_presentation.no_complete_training_audit_matched_the_current_16f96cb"
    )
    relevant_kinds = {"fold_membership", "tuning_trials", "model_comparison"}
    for log_path in candidates:
        try:
            if log_path.is_symlink() or not log_path.is_file():
                raise ValueError(_src("model_training_presentation.audit_path_is_not_safe_25def6f"))
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
                raise ValueError(
                    _src("model_training_presentation.training_audit_is_not_complete_7f296af")
                )
            if manifest.get("source_sha256") != source_sha256:
                raise ValueError(
                    _src(
                        "model_training_presentation.training_audit_source_fingerprint_does_not_match_0ce1057"
                    )
                )

            log_bytes = log_path.read_bytes()
            events = [json.loads(line) for line in log_bytes.splitlines() if line.strip()]
            if not events:
                raise ValueError(
                    _src("model_training_presentation.training_audit_jsonl_is_empty_ecce582")
                )
            if any(not isinstance(event, dict) for event in events):
                raise ValueError(
                    _src("model_training_presentation.training_audit_event_must_be_a_json_43010b2")
                )
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
                raise ValueError(
                    _src(
                        "model_training_presentation.training_audit_jsonl_run_identity_does_not_5a6692a"
                    )
                )

            downloads = [
                {
                    "label": _src("model_training_presentation.audit_manifest_13d499c"),
                    "kind": "audit_manifest",
                    "filename": f"training-{pipeline_run_id}-manifest.json",
                    "sha256": _sha256_bytes(manifest_bytes),
                    "content": manifest_bytes,
                    "mime": "application/json",
                },
                {
                    "label": _src(
                        "model_training_presentation.complete_structured_training_log_3a59f7e"
                    ),
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
                    raise ValueError(
                        _tr(
                            "model_training_presentation.training_audit_export_checksum_mismatch_value0_95c2d18",
                            value0=f"{export_path.name}",
                        )
                    )
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
            relevant_operations = {
                "temporal_cv_evaluation",
                "evaluate_rf_tuning_trial",
                "rank_rf_n_estimators",
                "record_applied_model_selection",
                "record_rf_manual_tuning_summary",
                "create_final_estimator",
                "score_locked_test",
                "predict_locked_test",
                "compute_locked_test_permutation_importance",
                "summarize_locked_test_absolute_error",
            }
            relevant_events = [
                event for event in events if event.get("operation") in relevant_operations
            ]
            event_preview = [
                {
                    "sequence": event.get("sequence"),
                    "timestamp": event.get("timestamp"),
                    "operation": event.get("operation"),
                    "status": event.get("status"),
                    "model": event.get("model") or event.get("model_name"),
                    "fold_id": event.get("fold_id") or event.get("fold"),
                    "trial_id": event.get("trial_id") or event.get("evaluation_id"),
                    "message": (
                        str(event["message"])[:500] if event.get("message") is not None else None
                    ),
                    "evidence_ref": f"{log_path.name}#sequence={event.get('sequence')}",
                }
                for event in relevant_events[:200]
            ]
            return {
                "available": True,
                "audit_run_id": audit_run_id,
                "pipeline_run_id": pipeline_run_id,
                "training_status": manifest["training_status"],
                "source_sha256": source_sha256,
                "selection_source": selection.get("selection_source"),
                "event_count": len(events),
                "relevant_event_count": len(relevant_events),
                "event_preview": event_preview,
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
        raise ValueError(
            _src("model_training_presentation.candidate_fold_evidence_is_incomplete_a76324e")
        )
    if membership.empty or not required_membership <= set(membership):
        raise ValueError(
            _src("model_training_presentation.fold_membership_evidence_is_incomplete_3c3f05e")
        )
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
            _tr("model_training_presentation.dev_rows_are_sorted_chronologically_and_split_616bd90")
        ),
        "limitations": (
            _tr("model_training_presentation.this_is_not_randomized_k_fold_and_415dfe7")
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
        raise ValueError(
            _src("model_training_presentation.initial_tuning_grid_evidence_is_incomplete_b0dbfca")
        )
    stage_counts: dict[str, int] = {}
    for parameter, stem in stage_columns:
        frame = tables.get(stem, pd.DataFrame())
        if frame.empty or parameter not in frame or "CV_R2" not in frame:
            raise ValueError(
                _tr(
                    "model_training_presentation.tuning_stage_evidence_is_incomplete_value0_7143d1e",
                    value0=f"{parameter}",
                )
            )
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
        "ranking_metric": _src("model_training_presentation.cv_r_c40ab4a"),
        "ranking_direction": _src("model_training_presentation.higher_is_better_8cde7e0"),
        "saved_matches_initial_winner": saved_matches,
        "applied_parameters": {name: applied_parameters.get(name) for name in compared},
        "selection_scope": _src(
            "model_training_presentation.inherited_non_nested_dev_temporal_validation_locked_f95dd6a"
        ),
    }


def load_evidence_download(root: Path, manifest: dict[str, Any], stem: str) -> dict[str, Any]:
    """Return byte-exact content for one validated active evidence file."""
    matches = [entry for entry in manifest["files"] if Path(entry["path"]).stem == stem]
    if len(matches) != 1:
        raise EvidenceContractError(
            _tr(
                "model_training_presentation.evidence_file_is_unavailable_or_ambiguous_value0_acb1c2b",
                value0=f"{stem}",
            )
        )
    entry = matches[0]
    workspace = root.resolve()
    path = root / entry["path"]
    if path.is_symlink():
        raise EvidenceContractError(
            _tr(
                "model_training_presentation.evidence_download_path_is_unsafe_value0_e898b48",
                value0=f"{entry['path']}",
            )
        )
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise EvidenceContractError(
            _tr(
                "model_training_presentation.evidence_download_is_unavailable_value0_1adff10",
                value0=f"{entry['path']}",
            )
        ) from exc
    if not resolved.is_relative_to(workspace):
        raise EvidenceContractError(
            _tr(
                "model_training_presentation.evidence_download_path_escapes_workspace_value0_603021a",
                value0=f"{entry['path']}",
            )
        )
    content = resolved.read_bytes()
    actual_sha256 = _sha256_bytes(content)
    if actual_sha256 != entry["sha256"]:
        raise EvidenceContractError(
            _tr(
                "model_training_presentation.evidence_download_checksum_mismatch_value0_fc10977",
                value0=f"{entry['path']}",
            )
        )
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
        return _src("model_training_presentation.none_dc937b5")
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
                _src("model_training_presentation.rank_a4130d7"),
                _src("model_training_presentation.model_5e2c614"),
                _src("model_training_presentation.decision_role_b8dd1f9"),
                _src("model_training_presentation.mean_validation_mae_6a465d9"),
                _src("model_training_presentation.mean_train_mae_58f5297"),
                _src("model_evidence.validation_r_538b15c"),
            ]
        )
    table = summary.copy()
    table["validation_MAE_mean"] = pd.to_numeric(table["validation_MAE_mean"], errors="coerce")
    table = table.dropna(subset=["validation_MAE_mean"]).sort_values(
        ["validation_MAE_mean", "model"], kind="stable"
    )
    table[_src("model_training_presentation.rank_a4130d7")] = (
        table["validation_MAE_mean"].rank(method="min").astype(int)
    )
    best = float(table["validation_MAE_mean"].min())
    rows: list[dict[str, Any]] = []
    for _, row in table.iterrows():
        is_winner = math.isclose(float(row.validation_MAE_mean), best, rel_tol=0.0, abs_tol=0.0)
        is_reference = row.model == _src("model_evidence.dummy_median_accc4f4")
        emphasis = "bold" if is_winner else "italic" if is_reference else None
        role = (
            _src("model_training_presentation.winner_105dc74")
            if is_winner
            else _src("model_evidence.reference_baseline_58e7d63")
            if is_reference
            else _src("model_training_presentation.candidate_b2452d1")
        )
        rows.append(
            {
                _src("model_training_presentation.rank_a4130d7"): int(row.Rank),
                _src("model_training_presentation.model_5e2c614"): _markdown_emphasis(
                    str(row.model), emphasis
                ),
                _src("model_training_presentation.decision_role_b8dd1f9"): _markdown_emphasis(
                    role, emphasis
                ),
                _src("model_training_presentation.mean_validation_mae_6a465d9"): _markdown_emphasis(
                    f"${float(row.validation_MAE_mean):,.0f}", emphasis
                ),
                _src("model_training_presentation.mean_train_mae_58f5297"): _markdown_emphasis(
                    f"${float(row.train_MAE_mean):,.0f}", emphasis
                ),
                _src("model_evidence.validation_r_538b15c"): _markdown_emphasis(
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
                _src("model_training_presentation.hyperparameter_5bc576c"),
                _src("model_training_presentation.search_space_ec5ba5c"),
                _src("model_training_presentation.stage_winner_6d40f65"),
                _src("model_training_presentation.applied_value_f7d2bbd"),
                _src("model_training_presentation.relationship_25490a1"),
                _src("model_training_presentation.best_cv_r_2dc7204"),
                _src("model_training_presentation.best_cv_mae_236adf4"),
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
                _src("model_training_presentation.hyperparameter_5bc576c"): parameter,
                _src("model_training_presentation.search_space_ec5ba5c"): f"*{row.search_space}*",
                _src("model_training_presentation.stage_winner_6d40f65"): _markdown_emphasis(
                    _display_parameter(winner), "bold"
                ),
                _src("model_training_presentation.applied_value_f7d2bbd"): _markdown_emphasis(
                    _display_parameter(applied), "bold"
                ),
                _src("model_training_presentation.relationship_25490a1"): (
                    _src("model_training_presentation.applied_stage_winner_203b9a2")
                    if matches
                    else _src("model_training_presentation.sensitivity_only_not_applied_cce45a7")
                ),
                _src("model_training_presentation.best_cv_r_2dc7204"): _markdown_emphasis(
                    f"{float(row.best_cv_r2):.3f}", "bold"
                ),
                _src("model_training_presentation.best_cv_mae_236adf4"): _markdown_emphasis(
                    f"${float(row.best_cv_mae):,.0f}", "bold"
                ),
            }
        )
    return pd.DataFrame(rows)
