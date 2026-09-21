"""Offline entrypoint for training-validation/v1."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor

from .core import MODEL_FEATURES, TARGET, candidate_models, make_model_pipeline
from .training_evaluation import (
    derive_performance_comparison,
    evaluate_frozen_models,
    extract_fitted_encoded_importance,
    extract_rf_fold_importance,
    fit_diagnostics,
    permutation_mae_importance,
    run_feature_family_ablation,
    score_frozen_pipelines,
    select_family,
    subgroup_metrics,
    summarize_candidates,
    uncertainty_evidence,
)
from .training_evidence_io import (
    EvidenceContractError,
    TrainingEventWriter,
    claim_holdout_access,
    complete_holdout_access,
    fail_holdout_access,
    holdout_identity,
    load_approval,
    load_policy,
    make_run_id,
    publish_staged_pack,
    safe_workspace_path,
    sha256_file,
    validate_complete_pack,
    write_json,
)
from .training_partitions import (
    FoldDeclaration,
    build_expanding_monthly_folds,
    build_partition_membership,
    monthly_row_counts,
    propose_partition,
    reconcile_exposure,
    summarize_folds,
    validate_fold_summary,
)
from .training_report import (
    build_agent_summary,
    build_fold_explanation,
    build_inner_fold_explanation,
    build_metric_catalog,
    build_model_conclusions,
    build_training_log,
    build_ui_summary,
    evaluate_outcome,
    render_report,
    write_report_charts,
)
from .training_runtime import (
    assess_operational_budgets,
    benchmark_bundle_load,
    benchmark_prediction,
    compute_accuracy_fit_pareto,
    summarize_runtime_samples,
)
from .training_search import (
    conditional_gridsearch_rf_search,
    evaluate_matched_variants,
    fit_final_variants,
)

MAX_CSV_BYTES = 100 * 1024 * 1024
MAX_CSV_ROWS = 100_000
_DEFAULT_POLICY = Path(__file__).resolve().parents[2] / "config/training_validation.json"


def _blocked(code: str, message: str, *, next_action: str) -> dict[str, Any]:
    return {
        "schema_version": "training-validation/inspect-v1",
        "execution_status": "blocked",
        "reason_code": code,
        "message": message,
        "next_action": next_action,
        "fits_performed": 0,
        "predictions_performed": 0,
    }


def _known_reserve_exposure(workspace: Path, reserve_months: set[str]) -> list[str]:
    references: list[str] = []
    historical = workspace / "outputs/02_data_ready_for_ml/locked_test_raw.csv"
    if historical.is_file() and not historical.is_symlink():
        try:
            dates = pd.read_csv(historical, usecols=["posting_year", "posting_month"])
        except (OSError, ValueError):
            return ["outputs/02_data_ready_for_ml/locked_test_raw.csv:unverifiable"]
        periods = {
            f"{int(year):04d}-{int(month):02d}"
            for year, month in zip(
                dates["posting_year"], dates["posting_month"], strict=True
            )
        }
        if periods & reserve_months:
            references.append("outputs/02_data_ready_for_ml/locked_test_raw.csv")
    return references


def inspect_workspace(
    workspace: Path | str, *, policy_path: Path | str | None = None
) -> dict[str, Any]:
    """Inspect temporal readiness without fitting, predicting, or writing."""
    try:
        root = Path(workspace).resolve(strict=True)
        policy = load_policy(Path(policy_path) if policy_path else _DEFAULT_POLICY)
        raw_path = safe_workspace_path(root, "outputs/01_data_basic_clean/basic_clean.csv")
        if not raw_path.is_file():
            return _blocked(
                "INPUT_MISSING",
                "The fixed cleaned training input is missing.",
                next_action="Run the separately approved data-preparation workflow.",
            )
        size = raw_path.stat().st_size
        if size > MAX_CSV_BYTES:
            return _blocked(
                "INPUT_RESOURCE_LIMIT",
                "The cleaned CSV exceeds the 100 MiB inspection limit.",
                next_action="Review the prepared dataset and resource policy.",
            )
        metadata = pd.read_csv(raw_path, usecols=["posting_year", "posting_month"])
        if len(metadata) > MAX_CSV_ROWS:
            return _blocked(
                "INPUT_RESOURCE_LIMIT",
                "The cleaned CSV exceeds the 100,000-row inspection limit.",
                next_action="Review the prepared dataset and resource policy.",
            )
        proposal = propose_partition(
            metadata,
            (
                policy["split_targets"]["train"],
                policy["split_targets"]["evaluation_holdout"],
                policy["split_targets"]["inference_reserve"],
            ),
        )
        dataset_id = sha256_file(raw_path)
        membership = build_partition_membership(
            metadata,
            dataset_id=dataset_id,
            train_end=proposal.train_end,
            holdout_end=proposal.holdout_end,
        )
        reserve_months = set(
            membership.loc[membership["partition"] == "INFERENCE_RESERVE", "period"]
        )
        known_refs = _known_reserve_exposure(root, reserve_months)
        attestation = "known-exposed" if known_refs else "unknown"
        exposure = reconcile_exposure(attestation, known_exposure_refs=known_refs)
        policy_digest = hashlib.sha256(
            json.dumps(policy, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        proposal_material = {
            "dataset_id": dataset_id,
            "policy_sha256": policy_digest,
            "train_end": proposal.train_end,
            "holdout_end": proposal.holdout_end,
            "actual_counts": dict(proposal.actual_counts),
            "actual_shares": proposal.actual_shares,
        }
        proposal_digest = hashlib.sha256(
            json.dumps(proposal_material, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        period_values = sorted(membership["period"].unique())
        reason_code = (
            "RESERVE_KNOWN_EXPOSED"
            if exposure.status == "known-exposed"
            else "RESERVE_EXPOSURE_UNKNOWN"
        )
        message = (
            "The proposed inference reserve contains historically exposed rows."
            if exposure.status == "known-exposed"
            else "The inference reserve has no verified unseen-data attestation."
        )
        return {
            "schema_version": "training-validation/inspect-v1",
            "execution_status": "blocked",
            "reason_code": reason_code,
            "message": message,
            "next_action": "Obtain eligible future prepared data and approve exact boundaries/provenance.",
            "dataset": {
                "relative_path": "outputs/01_data_basic_clean/basic_clean.csv",
                "sha256": dataset_id,
                "bytes": size,
                "rows": len(metadata),
                "period_min": period_values[0],
                "period_max": period_values[-1],
                "observed_months": period_values,
            },
            "proposal": {
                "proposal_sha256": proposal_digest,
                "policy_sha256": policy_digest,
                "train_end": proposal.train_end,
                "holdout_end": proposal.holdout_end,
                "actual_counts": dict(proposal.actual_counts),
                "actual_shares": proposal.actual_shares,
                "objective": proposal.objective,
                "reserve_months": sorted(reserve_months),
            },
            "exposure": {
                "status": exposure.status,
                "ready": exposure.ready,
                "evidence_refs": list(exposure.evidence_refs),
                "assurance_limit": "Local evidence cannot prove absence of external use.",
            },
            "fit_budget": {
                "maximum_fits": 720,
                "maximum_benchmark_predict_calls": 1784,
            },
            "fits_performed": 0,
            "predictions_performed": 0,
        }
    except (EvidenceContractError, OSError, ValueError) as error:
        return _blocked(
            "INPUT_VALIDATION_FAILED",
            str(error),
            next_action="Correct the named input or policy and inspect again.",
        )


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _partition_sha(membership: pd.DataFrame) -> str:
    records = membership[
        ["dataset_id", "row_id", "source_position", "period", "partition"]
    ].to_dict("records")
    return _canonical_sha(records)


def build_approval_template(
    workspace: Path | str, *, policy_path: Path | str | None = None
) -> dict[str, Any]:
    """Build review material; the caller must add reviewer/time/exposure attestation."""
    root = Path(workspace).resolve(strict=True)
    policy_file = Path(policy_path) if policy_path else _DEFAULT_POLICY
    policy = load_policy(policy_file)
    raw = safe_workspace_path(root, "outputs/01_data_basic_clean/basic_clean.csv")
    metadata = pd.read_csv(raw, usecols=["posting_year", "posting_month"])
    proposal = propose_partition(
        metadata,
        (
            policy["split_targets"]["train"],
            policy["split_targets"]["evaluation_holdout"],
            policy["split_targets"]["inference_reserve"],
        ),
    )
    source_sha = sha256_file(raw)
    membership = build_partition_membership(
        metadata,
        dataset_id=source_sha,
        train_end=proposal.train_end,
        holdout_end=proposal.holdout_end,
    )
    policy_sha = _canonical_sha(policy)
    proposal_material = {
        "dataset_id": source_sha,
        "policy_sha256": policy_sha,
        "train_end": proposal.train_end,
        "holdout_end": proposal.holdout_end,
        "actual_counts": dict(proposal.actual_counts),
        "actual_shares": proposal.actual_shares,
    }
    return {
        "schema_version": "training-validation-approval/v1",
        "source_sha256": source_sha,
        "policy_sha256": policy_sha,
        "proposal_sha256": _canonical_sha(proposal_material),
        "partition_sha256": _partition_sha(membership),
        "train_end": proposal.train_end,
        "holdout_end": proposal.holdout_end,
        "actual_counts": dict(proposal.actual_counts),
        "actual_shares": proposal.actual_shares,
    }


def _validate_run_approval(
    root: Path,
    approval: dict[str, Any],
    policy: dict[str, Any],
    raw_path: Path,
    policy_file: Path,
) -> tuple[pd.DataFrame, str]:
    template = build_approval_template(root, policy_path=policy_file)
    for field in (
        "source_sha256",
        "policy_sha256",
        "proposal_sha256",
        "partition_sha256",
        "train_end",
        "holdout_end",
        "actual_counts",
    ):
        if approval[field] != template[field]:
            raise EvidenceContractError(f"stale approval field: {field}")
    for name, expected in template["actual_shares"].items():
        if not np.isclose(float(approval["actual_shares"][name]), float(expected), atol=1e-12):
            raise EvidenceContractError("stale approval field: actual_shares")
    reserve_attestation = approval["exposure"].get("reserve")
    if reserve_attestation != "attested-unexposed":
        raise EvidenceContractError("reserve requires an attested-unexposed approval")
    metadata = pd.read_csv(raw_path, usecols=["posting_year", "posting_month"])
    membership = build_partition_membership(
        metadata,
        dataset_id=template["source_sha256"],
        train_end=approval["train_end"],
        holdout_end=approval["holdout_end"],
    )
    reserve_months = set(
        membership.loc[membership["partition"] == "INFERENCE_RESERVE", "period"]
    )
    exposure = reconcile_exposure(
        reserve_attestation,
        known_exposure_refs=_known_reserve_exposure(root, reserve_months),
    )
    if not exposure.ready:
        raise EvidenceContractError(f"inference reserve exposure is {exposure.status}")
    checkout = Path(__file__).resolve().parents[2]
    producer_paths = [
        checkout / "src/ai_job_market" / name
        for name in (
            "core.py",
            "training_evaluation.py",
            "training_evidence_io.py",
            "training_partitions.py",
            "training_report.py",
            "training_runtime.py",
            "training_search.py",
            "training_validation.py",
        )
    ]
    identity_paths = [*producer_paths, checkout / "pyproject.toml", checkout / "uv.lock"]
    producer_identity = {
        path.relative_to(checkout).as_posix(): sha256_file(path)
        for path in identity_paths
        if path.is_file() and not path.is_symlink()
    }
    experiment_sha = _canonical_sha(
        {
            "source": template["source_sha256"],
            "policy": template["policy_sha256"],
            "proposal": template["proposal_sha256"],
            "partition": template["partition_sha256"],
            "producer_identity": producer_identity,
        }
    )
    return membership, experiment_sha


def compare_training_partition(
    frame: pd.DataFrame,
    membership: pd.DataFrame,
    folds: list[FoldDeclaration],
    *,
    models: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the frozen five-family comparison without opening protected populations."""
    candidates = models or candidate_models(seed=42)
    expected = {
        "Dummy Median",
        "Linear Regression",
        "Ridge Regression",
        "Random Forest",
        "Gradient Boosting",
    }
    if set(candidates) != expected:
        raise ValueError("comparison requires exactly the five declared candidate models")
    evaluation = evaluate_frozen_models(frame, membership, MODEL_FEATURES, candidates, folds)
    if len(evaluation.fold_metrics) != 5 * len(folds):
        raise ValueError("comparison evidence is incomplete; family selection is blocked")
    summary = summarize_candidates(evaluation.fold_metrics, evaluation.oof_predictions)
    selection = select_family(summary)
    selection["frozen"] = True
    diagnostics = fit_diagnostics(evaluation.fold_metrics, gap_threshold=0.2)
    return {
        "evaluation": evaluation,
        "summary": summary,
        "selection": selection,
        "fit_diagnostics": diagnostics,
    }


def _load_authorized_frame(
    raw_path: Path, membership: pd.DataFrame, *, target_partitions: set[str]
) -> pd.DataFrame:
    frame = pd.read_csv(raw_path, usecols=lambda column: column != TARGET)
    partition_by_position = membership.set_index("source_position")["partition"].to_dict()
    targets = np.full(len(frame), np.nan, dtype=float)
    with raw_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or TARGET not in reader.fieldnames:
            raise EvidenceContractError(f"training input is missing target column: {TARGET}")
        for position, row in enumerate(reader):
            if position >= len(frame):
                raise EvidenceContractError("target rows exceed feature rows")
            if partition_by_position[position] in target_partitions:
                try:
                    targets[position] = float(row[TARGET])
                except (TypeError, ValueError) as error:
                    raise EvidenceContractError("authorized target value is not numeric") from error
    frame[TARGET] = targets
    return frame


def _json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return frame.astype(object).where(pd.notna(frame), None).to_dict("records")


def _fold_membership(folds: list[FoldDeclaration]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fold in folds:
        for role, row_ids in (
            ("train", fold.train_row_ids),
            ("validation", fold.validation_row_ids),
        ):
            rows.extend(
                {
                    "fold_id": fold.fold_id,
                    "scope": fold.scope,
                    "parent_fold_id": fold.parent_fold_id,
                    "search_id": fold.search_id,
                    "role": role,
                    "row_id": row_id,
                    "membership_hash": fold.membership_hash,
                }
                for row_id in row_ids
            )
    return pd.DataFrame(rows)


def _inner_fold_contexts(
    train_membership: pd.DataFrame, outer_folds: list[FoldDeclaration]
) -> tuple[dict[str, list[FoldDeclaration]], list[FoldDeclaration]]:
    contexts: dict[str, list[FoldDeclaration]] = {}
    all_inner: list[FoldDeclaration] = []
    for outer in outer_folds:
        parent = train_membership[train_membership["row_id"].isin(outer.train_row_ids)]
        folds = build_expanding_monthly_folds(
            parent,
            n_splits=3,
            scope="inner",
            parent_population_id=outer.membership_hash,
            parent_fold_id=outer.fold_id,
            search_id=f"rf-{outer.fold_id}",
        )
        contexts[outer.fold_id] = folds
        all_inner.extend(folds)
    final_folds = build_expanding_monthly_folds(
        train_membership,
        n_splits=3,
        scope="inner",
        parent_population_id="TRAIN",
        parent_fold_id="final-train",
        search_id="rf-final-train",
    )
    contexts["final-train"] = final_folds
    all_inner.extend(final_folds)
    return contexts, all_inner


def _fit_full_train_candidates(
    frame: pd.DataFrame,
    membership: pd.DataFrame,
    *,
    seed: int,
    models: dict[str, Any] | None = None,
) -> dict[str, Any]:
    train_positions = membership.loc[
        membership["partition"] == "TRAIN", "source_position"
    ].to_numpy(dtype=int)
    fitted: dict[str, Any] = {}
    for name, estimator in (models or candidate_models(seed)).items():
        pipeline = make_model_pipeline(clone(estimator), MODEL_FEATURES)
        pipeline.fit(
            frame.iloc[train_positions][MODEL_FEATURES],
            frame.iloc[train_positions][TARGET],
        )
        fitted[name] = pipeline
    return fitted


def _evaluate_variants(
    frame: pd.DataFrame,
    membership: pd.DataFrame,
    outer_folds: list[FoldDeclaration],
    *,
    selected_family: str,
    fold_params: dict[str, dict[str, Any]],
    seed: int,
) -> pd.DataFrame:
    by_fold = {fold.fold_id: fold for fold in outer_folds}

    def evaluate(
        variant: str, features: list[str], params: dict[str, Any], fold_id: str
    ) -> dict[str, Any]:
        if selected_family == "Random Forest":
            estimator = RandomForestRegressor(random_state=seed, n_jobs=1, **params)
        else:
            estimator = clone(candidate_models(seed)[selected_family])
        result = evaluate_frozen_models(
            frame,
            membership,
            features,
            {selected_family: estimator},
            [by_fold[fold_id]],
        )
        row = result.fold_metrics.iloc[0].to_dict()
        return {
            key: value
            for key, value in row.items()
            if key.startswith("train_")
            or key.startswith("validation_")
            or key.endswith("_wall_s")
            or key.endswith("_cpu_s")
        }

    rows = evaluate_matched_variants(
        fold_winners=fold_params,
        full_features=MODEL_FEATURES,
        top2_features=["job_category", "years_of_experience"],
        evaluate=evaluate,
    )
    return pd.DataFrame(rows)


def _write_training_pack(
    staging: Path,
    *,
    run_id: str,
    experiment_sha: str,
    approval: dict[str, Any],
    policy: dict[str, Any],
    membership: pd.DataFrame,
    all_folds: list[FoldDeclaration],
    outer_fold_summary: pd.DataFrame,
    all_fold_summary: pd.DataFrame,
    comparison: dict[str, Any],
    ablation: pd.DataFrame,
    fold_importance: pd.DataFrame,
    encoded_importance: pd.DataFrame,
    importance_drift: pd.DataFrame,
    tuning_trials: pd.DataFrame,
    tuning_summary: dict[str, Any],
    variant_metrics: pd.DataFrame,
    holdout_metrics: pd.DataFrame,
    holdout_predictions: pd.DataFrame,
    permutation: pd.DataFrame,
    subgroups: pd.DataFrame,
    uncertainty: list[dict[str, Any]],
    runtime_samples: pd.DataFrame,
    runtime_summary: pd.DataFrame,
    bundle_load_summary: pd.DataFrame,
    performance: pd.DataFrame,
    operational: dict[str, Any],
    outcome: dict[str, Any],
    fit_count: int,
) -> None:
    if not staging.is_dir():
        raise EvidenceContractError("run staging directory is unavailable")
    counts = membership["partition"].value_counts().to_dict()
    fold_text = build_fold_explanation(
        outer_fold_summary, partition_counts=counts
    ) + build_inner_fold_explanation(all_fold_summary)
    conclusions = build_model_conclusions(
        comparison["summary"],
        selection=comparison["selection"],
        final_metrics=holdout_metrics[
            holdout_metrics["model_role_id"].isin(["final_full", "final_top2"])
        ],
    )
    declaration = {
        "schema_version": "training-validation/v1",
        "run_id": run_id,
        "experiment_sha256": experiment_sha,
        "train_end": approval["train_end"],
        "holdout_end": approval["holdout_end"],
        "partition_sha256": approval["partition_sha256"],
        "counts": counts,
        "shares": approval["actual_shares"],
        "reserve_exposure": approval["exposure"]["reserve"],
    }
    evaluation_declaration = {
        "schema_version": "training-validation/v1",
        "run_id": run_id,
        "selected_family": comparison["selection"]["selected_family"],
        "selection": comparison["selection"],
        "final_variants": ["final_full", "final_top2"],
        "holdout_id": holdout_identity(
            tuple(
                membership.loc[
                    membership["partition"] == "EVALUATION_HOLDOUT", "row_id"
                ]
            )
        ),
    }
    run_context = {
        "who": f"local operator; reviewer {approval['reviewer']}",
        "what": "continuous annual salary regression with five candidate families",
        "when": f"approved {approval['approved_at']}",
        "where": f"outputs/training_validation/{run_id}",
        "why": "compare models without future leakage",
        "how": "whole-month partitions, expanding folds and one frozen holdout evaluation",
    }
    report = render_report(
        run_context=run_context,
        fold_explanation=fold_text,
        model_conclusions=conclusions,
    )
    derived = derive_performance_comparison(
        comparison["evaluation"].oof_predictions,
        variant_fold_metrics=variant_metrics,
        tuning_baseline_mae=None,
        tuning_selected_mae=None,
    )
    agent = build_agent_summary(
        run_id=run_id,
        summary=comparison["summary"],
        selection=comparison["selection"],
        fold_summary=all_fold_summary,
        conclusions=conclusions,
    )
    ui = build_ui_summary(
        run_id=run_id,
        manifest_sha256=None,
        summary=comparison["summary"],
        selection=comparison["selection"],
        fold_summary=all_fold_summary,
        conclusions=conclusions,
    )
    tables = {
        "partition_membership.csv": membership,
        "fold_membership.csv": _fold_membership(all_folds),
        "fold_summary.csv": all_fold_summary,
        "monthly_row_counts.csv": monthly_row_counts(membership),
        "oof_predictions.csv": comparison["evaluation"].oof_predictions,
        "candidate_fold_metrics.csv": comparison["evaluation"].fold_metrics,
        "candidate_summary.csv": comparison["summary"],
        "runtime.csv": comparison["evaluation"].fold_metrics[
            [column for column in comparison["evaluation"].fold_metrics if column.endswith("_s") or column in {"model", "fold_id"}]
        ],
        "fit_diagnostics.csv": comparison["fit_diagnostics"],
        "ablation_metrics.csv": ablation,
        "fold_importance.csv": fold_importance,
        "importance_drift.csv": importance_drift,
        "tuning_trials.csv": tuning_trials,
        "variant_fold_metrics.csv": variant_metrics,
        "holdout_metrics.csv": holdout_metrics,
        "holdout_predictions.csv": holdout_predictions,
        "encoded_importance.csv": encoded_importance,
        "permutation_importance.csv": permutation,
        "subgroup_metrics.csv": subgroups,
        "runtime_samples.csv": runtime_samples,
        "runtime_summary.csv": pd.concat(
            [runtime_summary, bundle_load_summary], ignore_index=True, sort=False
        ),
        "performance_comparison.csv": performance,
        "accuracy_runtime_tradeoff.csv": compute_accuracy_fit_pareto(comparison["summary"]),
    }
    for relative, table in tables.items():
        table.to_csv(staging / relative, index=False)
    write_json(staging / "partition_declaration.json", declaration)
    write_json(staging / "model_conclusions.json", conclusions)
    write_json(staging / "metric_catalog.json", build_metric_catalog())
    write_json(staging / "agent_summary.json", agent)
    write_json(staging / "ui_summary.json", ui)
    write_json(staging / "family_selection.json", comparison["selection"])
    write_json(staging / "tuning_summary.json", tuning_summary)
    write_json(staging / "evaluation_declaration.json", evaluation_declaration)
    write_json(staging / "uncertainty.json", uncertainty)
    write_json(staging / "operational_assessment.json", operational)
    write_json(
        staging / "conclusion.json",
        {
            "execution_status": "complete",
            "scientific_outcome": outcome,
            "selected_model_id": comparison["selection"]["selected_family"],
            "fit_count": fit_count,
            "limitations": [
                "Local evidence cannot prove absence of external data exposure.",
                "No deployment or UI activation is performed.",
            ],
            "next_action": "Human review of evidence and operational criteria.",
        },
    )
    (staging / "fold_explanation.md").write_text(fold_text, encoding="utf-8")
    (staging / "report.md").write_text(report, encoding="utf-8")
    training_log = build_training_log(
        run_context=run_context,
        requested_shares=policy["split_targets"],
        actual_counts=counts,
        actual_shares=approval["actual_shares"],
        fold_summary=all_fold_summary,
        candidate_fold_metrics=comparison["evaluation"].fold_metrics,
        candidate_summary=comparison["summary"],
        fit_diagnostic_rows=comparison["fit_diagnostics"],
        ablation=ablation,
        fold_importance=fold_importance,
        importance_drift=importance_drift,
        selection=comparison["selection"],
        tuning_trials=tuning_trials,
        tuning_summary=tuning_summary,
        variant_metrics=variant_metrics,
        holdout_metrics=holdout_metrics,
        holdout_predictions=holdout_predictions,
        encoded_importance=encoded_importance,
        permutation_importance=permutation,
        subgroups=subgroups,
        uncertainty=uncertainty,
        outcome=outcome,
        operational=operational,
        fit_count=fit_count,
    )
    (staging / "training.log").write_text(training_log, encoding="utf-8")
    importance_chart = (
        permutation.groupby("name", as_index=False)["mae_increase"].mean()
    )
    write_report_charts(
        staging / "charts",
        comparison["summary"],
        holdout_predictions=holdout_predictions[
            holdout_predictions["model_role_id"] == "final_full"
        ],
        importance=importance_chart,
    )
    if not (staging / "events.jsonl").exists():
        raise EvidenceContractError("events must be written before pack publication")
    if derived["models"].empty:
        raise EvidenceContractError("derived performance evidence is empty")


def run_workspace(
    workspace: Path | str,
    *,
    approval_path: Path | str,
    policy_path: Path | str | None = None,
    models: dict[str, Any] | None = None,
) -> dict[str, Any]:
    root = Path(workspace).resolve(strict=True)
    policy_source = Path(policy_path) if policy_path else _DEFAULT_POLICY
    if policy_source.is_symlink():
        raise EvidenceContractError("policy file cannot be a symlink")
    policy_file = policy_source.resolve(strict=True)
    checkout = Path(__file__).resolve().parents[2]
    if not (policy_file.is_relative_to(root) or policy_file.is_relative_to(checkout)):
        raise EvidenceContractError("policy path must stay inside workspace or checkout")
    approval_source = Path(approval_path).absolute()
    if any(
        candidate.is_symlink()
        for candidate in (approval_source, *approval_source.parents)
        if candidate.is_relative_to(root)
    ):
        raise EvidenceContractError("approval path cannot contain symlinks")
    approval_file = approval_source.resolve(strict=True)
    if not approval_file.is_relative_to(root):
        raise EvidenceContractError("approval path must stay inside the workspace")
    policy = load_policy(policy_file)
    approval = load_approval(approval_file)
    raw_path = safe_workspace_path(root, "outputs/01_data_basic_clean/basic_clean.csv")
    if raw_path.stat().st_size > MAX_CSV_BYTES:
        raise EvidenceContractError("cleaned CSV exceeds the 100 MiB run limit")
    membership, experiment_sha = _validate_run_approval(
        root, approval, policy, raw_path, policy_file
    )
    run_id = make_run_id(experiment_sha)
    final_dir = safe_workspace_path(
        root, Path("outputs/training_validation") / run_id
    )
    if final_dir.is_dir():
        manifest = validate_complete_pack(root, run_id)
        return {
            "execution_status": "reused",
            "run_id": run_id,
            "manifest": manifest,
            "fits_performed": 0,
            "predictions_performed": 0,
        }
    output_staging = safe_workspace_path(
        root, Path("outputs/training_validation") / f".{run_id}.staging"
    )
    artifact_final = safe_workspace_path(
        root, Path("artifacts/training_validation") / run_id
    )
    artifact_staging = safe_workspace_path(
        root, Path("artifacts/training_validation") / f".{run_id}.staging"
    )
    if output_staging.exists() or artifact_staging.exists() or artifact_final.exists():
        raise EvidenceContractError("incomplete or conflicting run namespace exists")
    started = time.perf_counter()
    holdout_id_value: str | None = None
    access_claimed = False
    try:
        output_staging.mkdir(parents=True)
        with TrainingEventWriter(output_staging / "events.jsonl", run_id) as events:
            operation = events.start("run", message="Authorized training validation started.")
            partition_operation = events.start(
                "partition_validation",
                message="Validating the approved chronological 80/19/1 partition and temporal folds.",
            )
            frame = _load_authorized_frame(
                raw_path, membership, target_partitions={"TRAIN"}
            )
            if len(frame) > MAX_CSV_ROWS:
                raise EvidenceContractError("cleaned CSV exceeds the 100,000-row run limit")
            required = set(MODEL_FEATURES) | {TARGET, "posting_year", "posting_month"}
            missing = sorted(required - set(frame.columns))
            if missing:
                raise EvidenceContractError(f"training input is missing columns: {missing}")
            numeric_features = frame[
                ["years_of_experience", "demand_score", "benefits_score_10", "skill_count"]
            ].apply(pd.to_numeric, errors="coerce")
            train_positions_for_validation = membership.loc[
                membership["partition"] == "TRAIN", "source_position"
            ].to_numpy(dtype=int)
            train_target = pd.to_numeric(
                frame.iloc[train_positions_for_validation][TARGET], errors="coerce"
            )
            if (
                numeric_features.isna().any().any()
                or not np.isfinite(numeric_features.to_numpy()).all()
                or train_target.isna().any()
                or not np.isfinite(train_target.to_numpy()).all()
            ):
                raise EvidenceContractError("training numeric values must be finite")
            if (numeric_features["years_of_experience"] < 0).any():
                raise EvidenceContractError("years_of_experience must be nonnegative")
            train_membership = membership[membership["partition"] == "TRAIN"]
            holdout_rows = int((membership["partition"] == "EVALUATION_HOLDOUT").sum())
            reserve_rows = int((membership["partition"] == "INFERENCE_RESERVE").sum())
            outer_folds = build_expanding_monthly_folds(
                train_membership,
                n_splits=5,
                scope="outer",
                parent_population_id=approval["partition_sha256"],
            )
            fold_summary = summarize_folds(
                outer_folds,
                parent_membership=train_membership,
                holdout_rows=holdout_rows,
                reserve_rows=reserve_rows,
            )
            validate_fold_summary(fold_summary, outer_folds, train_membership)
            inner_contexts, inner_folds = _inner_fold_contexts(train_membership, outer_folds)
            inner_summaries: list[pd.DataFrame] = []
            for context_id, context_folds in inner_contexts.items():
                if context_id == "final-train":
                    parent = train_membership
                else:
                    outer = next(fold for fold in outer_folds if fold.fold_id == context_id)
                    parent = train_membership[
                        train_membership["row_id"].isin(outer.train_row_ids)
                    ]
                context_summary = summarize_folds(
                    context_folds,
                    parent_membership=parent,
                    holdout_rows=holdout_rows,
                    reserve_rows=reserve_rows,
                )
                validate_fold_summary(context_summary, context_folds, parent)
                inner_summaries.append(context_summary)
            all_fold_summary = pd.concat(
                [fold_summary, *inner_summaries], ignore_index=True
            )
            events.complete(
                partition_operation,
                message="Chronological partition and expanding monthly folds validated.",
                details={
                    "requested_shares": policy["split_targets"],
                    "actual_counts": membership["partition"].value_counts().to_dict(),
                    "actual_shares": approval["actual_shares"],
                    "outer_fold_count": len(outer_folds),
                    "inner_fold_count": len(inner_folds),
                    "reserve_used": False,
                },
            )
            comparison_operation = events.start(
                "candidate_comparison",
                message="Training five frozen model families on identical outer folds.",
                planned_units=25,
            )
            comparison = compare_training_partition(
                frame, membership, outer_folds, models=models
            )
            seed = int(policy["seed"])
            rf_base = (models or candidate_models(seed))["Random Forest"]
            rf_rows = comparison["evaluation"].fold_metrics.query("model == 'Random Forest'")
            _, ablation = run_feature_family_ablation(
                frame,
                membership,
                outer_folds,
                features=MODEL_FEATURES,
                families=policy["feature_families"],
                model=clone(rf_base),
                baseline_fold_metrics=rf_rows,
            )
            _, family_importance, drift = extract_rf_fold_importance(
                comparison["evaluation"],
                features=MODEL_FEATURES,
                families=policy["feature_families"],
            )
            events.complete(
                comparison_operation,
                message="Five-model comparison, fit diagnosis, ablation, and fold importance completed.",
                details={
                    "candidate_folds": len(comparison["evaluation"].fold_metrics),
                    "selected_family": comparison["selection"]["selected_family"],
                    "lowest_cv_mae_model": comparison["selection"]["lowest_mae_model"],
                    "selection_method": comparison["selection"]["selection_method"],
                },
            )

            searches = {}
            for context_id, context_folds in inner_contexts.items():
                selected_family = comparison["selection"]["selected_family"]
                if selected_family != "Random Forest":
                    skip_operation = events.start(
                        "grid_search_skip",
                        message=f"GridSearchCV eligibility checked for {context_id}.",
                    )
                    searches[context_id] = conditional_gridsearch_rf_search(
                        selected_family=selected_family,
                        search_id=f"rf-{context_id}",
                        frame=frame,
                        membership=membership,
                        features=MODEL_FEATURES,
                        folds=context_folds,
                        policy=policy["rf_search"],
                        seed=seed,
                    )
                    events.complete(
                        skip_operation,
                        message=f"GridSearchCV skipped for {context_id} because Random Forest was not selected.",
                        details={
                            "context_id": context_id,
                            "selected_family": selected_family,
                            "reason": searches[context_id].reason,
                        },
                    )
                    continue
                grid_operation = events.start(
                    "grid_search",
                    message=f"GridSearchCV started for {context_id}.",
                    planned_units=len(policy["rf_search"]["n_estimators"])
                    * len(policy["rf_search"]["max_depth"]),
                )

                def log_grid_candidate(row: dict[str, Any], *, context: str = context_id) -> None:
                    candidate_operation = events.start(
                        "grid_search_candidate",
                        message=f"GridSearchCV candidate evaluated for {context}.",
                    )
                    events.complete(
                        candidate_operation,
                        message=f"GridSearchCV candidate completed for {context}.",
                        details={
                            "context_id": context,
                            "trial_id": row["trial_id"],
                            "params": row["params"],
                            "rank": row["rank"],
                            "mae_mean": row["mae_mean"],
                            "r2_mean": row["r2_mean"],
                            "status": row["status"],
                        },
                    )

                searches[context_id] = conditional_gridsearch_rf_search(
                    selected_family=selected_family,
                    search_id=f"rf-{context_id}",
                    frame=frame,
                    membership=membership,
                    features=MODEL_FEATURES,
                    folds=context_folds,
                    policy=policy["rf_search"],
                    seed=seed,
                    on_step=log_grid_candidate,
                )
                events.complete(
                    grid_operation,
                    message=f"GridSearchCV completed for {context_id}.",
                    details={
                        "context_id": context_id,
                        "inner_fold_ids": [fold.fold_id for fold in context_folds],
                        "scoring": policy["rf_search"]["scoring"],
                        "status": searches[context_id].status,
                        "winner": searches[context_id].winner,
                    },
                )
            if any(result.status == "failed" for result in searches.values()):
                raise EvidenceContractError("GridSearchCV did not produce a valid Random Forest candidate")
            trial_tables = [result.trials for result in searches.values() if not result.trials.empty]
            tuning_trials = pd.concat(trial_tables, ignore_index=True) if trial_tables else pd.DataFrame(columns=["search_id", "status"])
            tuning_summary = {
                context: {
                    "status": result.status,
                    "reason": result.reason,
                    "winner": result.winner,
                    "actual_fit_count": result.actual_fit_count,
                }
                for context, result in searches.items()
            }
            final_evaluation_operation = events.start(
                "final_evaluation",
                message="Fitting frozen Full and Top-2 variants before one-time holdout evaluation.",
            )
            selected = comparison["selection"]["selected_family"]
            fold_params = {
                fold.fold_id: (searches[fold.fold_id].winner or {})
                for fold in outer_folds
            }
            variant_metrics = _evaluate_variants(
                frame,
                membership,
                outer_folds,
                selected_family=selected,
                fold_params=fold_params,
                seed=seed,
            )
            final_params = searches["final-train"].winner or {}
            final_fits = fit_final_variants(
                frame,
                membership,
                selected_family=selected,
                params=final_params,
                full_features=MODEL_FEATURES,
                top2_features=policy["top2_features"],
                seed=seed,
            )
            full_train_candidates = _fit_full_train_candidates(
                frame, membership, seed=seed, models=models
            )
            artifact_staging.mkdir(parents=True)
            bundle_metadata: dict[str, Any] = {}
            for role, pipeline in final_fits.pipelines.items():
                name = "full" if role == "final_full" else "top2"
                bundle = artifact_staging / f"{name}.joblib"
                joblib.dump(pipeline, bundle)
                metadata = {
                    "run_id": run_id,
                    "model_role_id": role,
                    "selected_family": selected,
                    "features": MODEL_FEATURES if role == "final_full" else policy["top2_features"],
                    "path": f"artifacts/training_validation/{run_id}/{name}.joblib",
                    "metadata_path": f"artifacts/training_validation/{run_id}/{name}.metadata.json",
                    "sha256": sha256_file(bundle),
                    "bytes": bundle.stat().st_size,
                }
                write_json(artifact_staging / f"{name}.metadata.json", metadata)
                bundle_metadata[role] = metadata
            evaluation_declaration = {
                "selected_family": selected,
                "final_params": final_params,
                "partition_sha256": approval["partition_sha256"],
                "bundle_sha256": {role: value["sha256"] for role, value in bundle_metadata.items()},
            }
            declaration_sha = _canonical_sha(evaluation_declaration)
            holdout_ids = tuple(
                membership.loc[membership["partition"] == "EVALUATION_HOLDOUT", "row_id"]
            )
            holdout_id_value = holdout_identity(holdout_ids)
            access = claim_holdout_access(root, holdout_id_value, declaration_sha, run_id)
            if access.get("reuse"):
                raise EvidenceContractError("matching holdout evidence exists without a reusable run pack")
            access_claimed = True
            frame = _load_authorized_frame(
                raw_path,
                membership,
                target_partitions={"TRAIN", "EVALUATION_HOLDOUT"},
            )
            all_pipelines = full_train_candidates | final_fits.pipelines
            features_by_model = {name: MODEL_FEATURES for name in full_train_candidates}
            features_by_model |= {
                "final_full": MODEL_FEATURES,
                "final_top2": policy["top2_features"],
            }
            final_encoded_importance = pd.concat(
                [
                    extract_fitted_encoded_importance(
                        final_fits.pipelines["final_full"],
                        features=MODEL_FEATURES,
                        model_role_id="final_full",
                    ),
                    extract_fitted_encoded_importance(
                        final_fits.pipelines["final_top2"],
                        features=policy["top2_features"],
                        model_role_id="final_top2",
                    ),
                ],
                ignore_index=True,
            )
            holdout_metrics, holdout_predictions = score_frozen_pipelines(
                frame,
                membership,
                pipelines=all_pipelines,
                features_by_model=features_by_model,
            )
            holdout_positions = membership.loc[
                membership["partition"] == "EVALUATION_HOLDOUT", "source_position"
            ].to_numpy(dtype=int)
            holdout_frame = frame.iloc[holdout_positions]
            permutation = permutation_mae_importance(
                final_fits.pipelines["final_full"],
                holdout_frame,
                features=MODEL_FEATURES,
                families=policy["feature_families"],
                repeats=policy["permutation_repeats"],
                seed=seed,
            )
            subgroups = subgroup_metrics(
                holdout_predictions[
                    holdout_predictions["model_role_id"].isin(["final_full", "final_top2"])
                ],
                small_support=policy["small_subgroup_rows"],
            )
            uncertainty: list[dict[str, Any]] = []
            for role in ("final_full", "final_top2"):
                evidence, _ = uncertainty_evidence(
                    holdout_predictions[holdout_predictions["model_role_id"] == role]
                )
                uncertainty.append(evidence)
            benchmark_contexts: dict[str, tuple[Any, pd.DataFrame]] = {}
            for (model_name, fold_id), pipeline in comparison["evaluation"].fitted_pipelines.items():
                fold = next(item for item in outer_folds if item.fold_id == fold_id)
                positions = membership.set_index("row_id").loc[list(fold.train_row_ids), "source_position"].to_numpy(dtype=int)
                benchmark_contexts[f"{model_name}:{fold_id}"] = (
                    pipeline,
                    frame.iloc[positions][MODEL_FEATURES],
                )
            train_positions = train_membership["source_position"].to_numpy(dtype=int)
            benchmark_contexts["final_full"] = (
                final_fits.pipelines["final_full"],
                frame.iloc[train_positions][MODEL_FEATURES],
            )
            benchmark_contexts["final_top2"] = (
                final_fits.pipelines["final_top2"],
                frame.iloc[train_positions][policy["top2_features"]],
            )
            runtime_samples, benchmark_meta = benchmark_fitted_contexts(
                benchmark_contexts, environment_id="local-offline"
            )
            runtime_summary = summarize_runtime_samples(runtime_samples)
            artifact_staging.parent.mkdir(parents=True, exist_ok=True)
            bundle_load_rows = []
            for name in ("full", "top2"):
                bundle_path = artifact_staging / f"{name}.joblib"
                evidence, _ = benchmark_bundle_load(bundle_path, loader=joblib.load)
                loaded_pipeline = joblib.load(bundle_path)
                loaded_features = MODEL_FEATURES if name == "full" else policy["top2_features"]
                first_predict_started = time.perf_counter()
                loaded_pipeline.predict(frame.iloc[train_positions[:1]][loaded_features])
                evidence["first_loaded_predict_ms"] = (
                    time.perf_counter() - first_predict_started
                ) * 1000
                evidence["model_id"] = f"final_{name}"
                evidence["method_version"] = "bundle-load/v1"
                bundle_load_rows.append(evidence)
            bundle_load_summary = pd.DataFrame(bundle_load_rows)
            performance = compute_accuracy_fit_pareto(comparison["summary"])
            performance = performance.merge(
                derive_performance_comparison(
                    comparison["evaluation"].oof_predictions,
                    variant_fold_metrics=variant_metrics,
                    tuning_baseline_mae=None,
                    tuning_selected_mae=None,
                )["models"],
                on="model",
                how="left",
            )
            full_p95 = runtime_summary.query("model_id == 'final_full' and batch_size == 1")["p95_ms"].iloc[0]
            top2_p95 = runtime_summary.query("model_id == 'final_top2' and batch_size == 1")["p95_ms"].iloc[0]
            outcome = evaluate_outcome(
                holdout_metrics.query("model_role_id == 'final_full'").iloc[0].to_dict(),
                approval,
            )
            operational = assess_operational_budgets(
                {
                    "run_wall_s": time.perf_counter() - started,
                    "final_full_batch1_p95_ms": float(full_p95),
                    "final_top2_batch1_p95_ms": float(top2_p95),
                    "full_bundle_bytes": bundle_metadata["final_full"]["bytes"],
                    "top2_bundle_bytes": bundle_metadata["final_top2"]["bytes"],
                },
                limits=approval,
                scientific_outcome=outcome["status"],
            )
            events.complete(
                final_evaluation_operation,
                message="Frozen variant, holdout, explainability, subgroup, uncertainty, and runtime evidence completed.",
                details={
                    "selected_family": selected,
                    "holdout_rows": holdout_rows,
                    "reserve_used": False,
                    "scientific_status": outcome["status"],
                    "operational_status": operational["operational_status"],
                },
            )
            fit_count = 25 + 30 + sum(item.actual_fit_count for item in searches.values()) + 10 + 2 + 5
            if fit_count > 720 or benchmark_meta["benchmark_predict_calls"] + 2 > 1784:
                raise EvidenceContractError("declared fit/prediction budget was exceeded")
            _write_training_pack(
                output_staging,
                run_id=run_id,
                experiment_sha=experiment_sha,
                approval=approval,
                policy=policy,
                membership=membership,
                all_folds=outer_folds + inner_folds,
                outer_fold_summary=fold_summary,
                all_fold_summary=all_fold_summary,
                comparison=comparison,
                ablation=ablation,
                fold_importance=family_importance,
                encoded_importance=final_encoded_importance,
                importance_drift=drift,
                tuning_trials=tuning_trials,
                tuning_summary=tuning_summary,
                variant_metrics=variant_metrics,
                holdout_metrics=holdout_metrics,
                holdout_predictions=holdout_predictions,
                permutation=permutation,
                subgroups=subgroups,
                uncertainty=uncertainty,
                runtime_samples=runtime_samples,
                runtime_summary=runtime_summary,
                bundle_load_summary=bundle_load_summary,
                performance=performance,
                operational=operational,
                outcome=outcome,
                fit_count=fit_count,
            )
            events.complete(
                operation,
                message="Authorized training validation completed.",
                details={
                    "fit_count": fit_count,
                    "warm_context_predict_calls": 1782,
                    "first_loaded_predict_calls": 2,
                },
            )
        artifact_staging.rename(artifact_final)
        holdout_file = output_staging / "holdout_metrics.csv"
        complete_holdout_access(
            root,
            holdout_id_value,
            evidence_path=f"outputs/training_validation/{run_id}/holdout_metrics.csv",
            evidence_sha256=sha256_file(holdout_file),
        )
        publish_staged_pack(
            root,
            run_id,
            output_staging,
            metadata={
                "experiment_sha256": experiment_sha,
                "source_sha256": approval["source_sha256"],
                "policy_sha256": approval["policy_sha256"],
                "proposal_sha256": approval["proposal_sha256"],
                "partition_sha256": approval["partition_sha256"],
                "seed": policy["seed"],
                "requested_ratios": policy["split_targets"],
                "actual_counts": approval["actual_counts"],
                "actual_shares": approval["actual_shares"],
                "selected_family": selected,
                "scientific_outcome": outcome,
                "operational_assessment": operational,
                "fit_count": fit_count,
                "maximum_fit_count": 720,
                "benchmark_predict_calls": 1784,
                "artifact_bundles": bundle_metadata,
            },
        )
        return {
            "execution_status": "complete",
            "run_id": run_id,
            "fits_performed": fit_count,
            "predictions_performed": 1784,
        }
    except BaseException:
        if access_claimed and holdout_id_value is not None:
            try:
                fail_holdout_access(root, holdout_id_value, reason_code="RUN_FAILED")
            except EvidenceContractError:
                pass
        shutil.rmtree(output_staging, ignore_errors=True)
        shutil.rmtree(artifact_staging, ignore_errors=True)
        raise


def benchmark_fitted_contexts(
    contexts: dict[str, tuple[Any, pd.DataFrame]], *, environment_id: str
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Benchmark the 25 fitted candidate-fold and two fitted final contexts."""
    if len(contexts) != 27 or not {"final_full", "final_top2"} <= set(contexts):
        raise ValueError("benchmark requires 25 candidate-fold and two final contexts")
    frames: list[pd.DataFrame] = []
    calls = 0
    for context_id, (pipeline, features) in contexts.items():
        if features.empty or TARGET in features.columns:
            raise ValueError("benchmark inputs must be nonempty feature-only TRAIN rows")
        samples, metadata = benchmark_prediction(
            pipeline,
            features,
            model_id=context_id,
            fold_id=context_id,
            environment_id=environment_id,
            batch_sizes=[1, 100],
            warmups=3,
            measured_calls=30,
        )
        frames.append(samples)
        calls += int(metadata["benchmark_predict_calls"])
    if calls != 1782:
        raise ValueError("benchmark call accounting violated the declared protocol")
    return pd.concat(frames, ignore_index=True), {
        "benchmark_predict_calls": calls,
        "remaining_first_load_predict_calls": 2,
        "maximum_predict_calls": 1784,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline training-validation evidence")
    subparsers = parser.add_subparsers(dest="command", required=True)
    inspect_parser = subparsers.add_parser("inspect", help="Inspect data readiness without fitting")
    inspect_parser.add_argument("--workspace", type=Path, required=True)
    inspect_parser.add_argument("--policy", type=Path)
    run_parser = subparsers.add_parser("run", help="Run an explicitly approved experiment")
    run_parser.add_argument("--workspace", type=Path, required=True)
    run_parser.add_argument("--approval", type=Path, required=True)
    run_parser.add_argument("--policy", type=Path)
    check_parser = subparsers.add_parser("check", help="Validate an immutable evidence pack")
    check_parser.add_argument("--workspace", type=Path, required=True)
    check_parser.add_argument("--run-id", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "inspect":
        result = inspect_workspace(args.workspace, policy_path=args.policy)
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        print(
            f"Inspection status: {result['execution_status']}. {result['message']} "
            "No fitting or prediction was performed. The inference reserve remains protected.",
            file=sys.stderr,
        )
        return 0 if result["execution_status"] == "ready" else 3
    if args.command == "run":
        try:
            result = run_workspace(
                args.workspace,
                approval_path=args.approval,
                policy_path=args.policy,
            )
        except EvidenceContractError as error:
            payload = _blocked(
                "RUN_BLOCKED",
                str(error),
                next_action="Correct the named approval/data condition and inspect again.",
            )
            print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
            print(f"Training validation blocked: {error}", file=sys.stderr)
            return 3
        except (OSError, ValueError, RuntimeError) as error:
            payload = _blocked(
                "RUN_FAILED",
                str(error),
                next_action="Inspect the failed stage before a new approved experiment.",
            )
            print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
            print("Training validation failed safely.", file=sys.stderr)
            return 4
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        print(f"Training validation status: {result['execution_status']}.", file=sys.stderr)
        return 0
    if args.command == "check":
        try:
            manifest = validate_complete_pack(args.workspace, args.run_id)
        except EvidenceContractError as error:
            print(f"Evidence check failed: {error}", file=sys.stderr)
            return 4
        print(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))
        print("Evidence pack checksums and required files are valid.", file=sys.stderr)
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
