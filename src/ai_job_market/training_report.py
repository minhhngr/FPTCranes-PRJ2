"""Pure human, agent, and future-UI views over validated training evidence."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go

CANDIDATE_ROLES = (
    "Dummy Median",
    "Linear Regression",
    "Ridge Regression",
    "Random Forest",
    "Gradient Boosting",
)
FINAL_ROLES = ("final_full", "final_top2")


def build_metric_catalog() -> dict[str, dict[str, Any]]:
    return {
        "mae_usd": {
            "name": "Mean absolute error",
            "definition": "Mean absolute value of actual minus predicted annual salary.",
            "unit": "USD/year",
            "direction": "lower",
            "display_precision": 0,
        },
        "rmse_usd": {
            "name": "Root mean squared error",
            "definition": "Square root of mean squared annual-salary residual.",
            "unit": "USD/year",
            "direction": "lower",
            "display_precision": 0,
        },
        "medae_usd": {
            "name": "Median absolute error",
            "definition": "Median absolute annual-salary residual.",
            "unit": "USD/year",
            "direction": "lower",
            "display_precision": 0,
        },
        "r2": {
            "name": "R-squared",
            "definition": "One minus residual sum of squares divided by total sum of squares.",
            "unit": "unitless",
            "direction": "higher",
            "display_precision": 3,
            "null_reasons": ["fewer_than_two_rows", "constant_target"],
        },
        "fit_wall_s": {
            "name": "Pipeline fit wall time",
            "definition": "Elapsed wall time for preprocessing and estimator fit.",
            "unit": "seconds",
            "direction": "context_only",
            "display_precision": 3,
        },
        "classification_metrics": {
            "name": "Classification metric family",
            "definition": "Accuracy, precision, recall, F1 and AUC require categorical targets and are educational only here.",
            "unit": "not_applicable",
            "direction": "not_applicable",
            "applicability": "continuous_salary_regression",
        },
    }


def build_fold_explanation(
    fold_summary: pd.DataFrame, *, partition_counts: dict[str, int]
) -> str:
    required = {
        "fold_id",
        "ordinal",
        "train_period_min",
        "train_period_max",
        "validation_period_min",
        "train_rows",
        "validation_rows",
        "not_used_yet_rows",
        "row_overlap_count",
        "shared_month_count",
        "chronological_order_ok",
        "expanding_history_ok",
        "holdout_rows_excluded",
        "reserve_rows_excluded",
        "evidence_ref",
    }
    missing = sorted(required - set(fold_summary.columns))
    if missing:
        raise ValueError(f"fold summary is missing fields: {missing}")
    if len(fold_summary) != 5:
        raise ValueError("outer fold explanation requires exactly five folds")
    total = sum(partition_counts.values())
    lines = [
        "# How expanding monthly folds are split",
        "",
        (
            "CV splits only the TRAIN partition. The later evaluation holdout is not used "
            "for tuning. The later inference reserve is not evaluated."
        ),
        "",
        "## Outer partition counts",
    ]
    for name in ("TRAIN", "EVALUATION_HOLDOUT", "INFERENCE_RESERVE"):
        count = int(partition_counts[name])
        lines.append(f"- {name}: {count} rows ({100 * count / total:.2f}% of {total}).")
    lines.extend(["", "## Five outer folds"])
    previous_train_rows: int | None = None
    for _, row in fold_summary.sort_values("ordinal").iterrows():
        expansion = (
            "not applicable for initial history"
            if previous_train_rows is None
            else f"training grew by {int(row.train_rows) - previous_train_rows} rows"
        )
        lines.extend(
            [
                "",
                f"### Outer fold {int(row.ordinal)}/5 — expanding monthly validation",
                (
                    f"Train: {int(row.train_rows)} rows ({row.train_period_min} to "
                    f"{row.train_period_max})."
                ),
                f"Validate: {int(row.validation_rows)} rows in {row.validation_period_min}.",
                f"Not used yet inside TRAIN: {int(row.not_used_yet_rows)} later rows.",
                f"Change from previous fold: {expansion}.",
                (
                    f"Checks: {int(row.row_overlap_count)} shared rows; "
                    f"{int(row.shared_month_count)} shared months; training strictly earlier: "
                    f"{'pass' if row.chronological_order_ok else 'fail'}; expanding history: "
                    f"{row.expanding_history_ok if row.expanding_history_ok is not None else 'not applicable'}."
                ),
                (
                    f"The separate holdout ({int(row.holdout_rows_excluded)} rows) and "
                    f"inference reserve ({int(row.reserve_rows_excluded)} rows) are excluded."
                ),
                "Conclusion: this fold keeps all observed earlier months and validates the next declared month.",
                f"Evidence: {row.evidence_ref}.",
            ]
        )
        previous_train_rows = int(row.train_rows)
    lines.extend(
        [
            "",
            (
                "A previous validation month can join a later fold's training history because it is then "
                "in the past. This is expanding validation, not randomized K-fold and not a sliding "
                "fixed-row window."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def build_inner_fold_explanation(fold_summary: pd.DataFrame) -> str:
    inner = fold_summary[fold_summary["scope"] == "inner"].copy()
    if inner.empty:
        return ""
    required = {
        "fold_id",
        "parent_fold_id",
        "search_id",
        "ordinal",
        "parent_population_rows",
        "train_period_min",
        "train_period_max",
        "validation_period_min",
        "train_rows",
        "validation_rows",
        "not_used_yet_rows",
    }
    missing = sorted(required - set(inner.columns))
    if missing:
        raise ValueError(f"inner fold summary is missing fields: {missing}")
    lines = [
        "## Parent-labelled inner tuning folds",
        "",
        "These three-fold expanding splits stay inside each outer-training parent (plus final TRAIN).",
    ]
    for _, row in inner.sort_values(["parent_fold_id", "ordinal"]).iterrows():
        lines.append(
            f"- {row.fold_id} (search {row.search_id}; parent {row.parent_fold_id}, "
            f"{int(row.parent_population_rows)} rows): train {int(row.train_rows)} rows "
            f"from {row.train_period_min} to {row.train_period_max}; validate "
            f"{int(row.validation_rows)} rows in {row.validation_period_min}; "
            f"{int(row.not_used_yet_rows)} parent rows remain later and unused."
        )
    return "\n".join(lines) + "\n"


def _candidate_conclusion(
    role: str, row: pd.Series | None, selection: dict[str, Any]
) -> dict[str, Any]:
    if row is None:
        return {
            "model_role_id": role,
            "status": "unavailable",
            "finding": "Required candidate evidence is missing.",
            "accuracy_evidence": None,
            "fit_diagnosis": "unavailable",
            "stability_evidence": None,
            "runtime_evidence": None,
            "decision": "not_eligible_missing_evidence",
            "limitation": "Missing fold evidence prevents a model conclusion.",
            "next_action": "Repair candidate evidence inside TRAIN before selection.",
            "evidence_refs": [f"candidate_summary.csv#model={role}"],
        }
    if role == selection.get("selected_family"):
        decision = "selected_family"
    elif role == selection.get("lowest_mae_model"):
        decision = "lowest_cv_mae_not_selected"
    else:
        decision = "not_selected_by_frozen_rule"
    fit = "baseline" if role == "Dummy Median" else "see_fit_diagnostics"
    return {
        "model_role_id": role,
        "configuration_id": row.configuration_id,
        "status": "available",
        "finding": f"Mean outer-validation MAE is {float(row.cv_mae_mean_usd):.2f} USD/year.",
        "accuracy_evidence": {
            "cv_mae_mean_usd": float(row.cv_mae_mean_usd),
            "cv_mae_sd_usd": float(row.cv_mae_sd_usd),
            "cv_rmse_mean_usd": float(row.cv_rmse_mean_usd),
            "cv_medae_mean_usd": float(row.cv_medae_mean_usd),
            "cv_r2_mean": None if pd.isna(row.cv_r2_mean) else float(row.cv_r2_mean),
        },
        "fit_diagnosis": fit,
        "stability_evidence": {
            "valid_folds": int(row.valid_fold_count),
            "failed_folds": int(row.failed_fold_count),
            "worst_fold_id": row.worst_fold_id,
        },
        "runtime_evidence": {"total_cv_fit_wall_s": float(row.total_cv_fit_wall_s)},
        "decision": decision,
        "selection_or_comparison_reason": selection.get("selection_method", "frozen_rule"),
        "limitation": "Five temporal folds are descriptive and do not establish production fitness.",
        "next_action": (
            "Proceed to the declared conditional tuning/evaluation stage."
            if decision == "selected_family"
            else "Retain this result as comparison evidence; do not tune it using the holdout."
        ),
        "evidence_refs": [f"candidate_summary.csv#model={role}"],
    }


def build_model_conclusions(
    summary: pd.DataFrame,
    *,
    selection: dict[str, Any],
    final_metrics: pd.DataFrame,
) -> list[dict[str, Any]]:
    by_model = {str(row.model): row for _, row in summary.iterrows()}
    conclusions = [
        _candidate_conclusion(role, by_model.get(role), selection) for role in CANDIDATE_ROLES
    ]
    by_final = {
        str(row.model_role_id): row for _, row in final_metrics.iterrows()
    } if not final_metrics.empty else {}
    for role in FINAL_ROLES:
        row = by_final.get(role)
        if row is None:
            conclusions.append(
                {
                    "model_role_id": role,
                    "status": "unavailable",
                    "finding": "Final holdout evidence is unavailable.",
                    "accuracy_evidence": None,
                    "fit_diagnosis": "not_assessed",
                    "stability_evidence": None,
                    "runtime_evidence": None,
                    "decision": "comparison_only_no_automatic_promotion",
                    "limitation": "A frozen, authorized holdout evaluation is missing.",
                    "next_action": "Do not infer final performance or open protected data.",
                    "evidence_refs": [f"holdout_metrics.csv#model_role_id={role}"],
                }
            )
            continue
        conclusions.append(
            {
                "model_role_id": role,
                "status": "available",
                "finding": f"Holdout MAE is {float(row.MAE):.2f} USD/year.",
                "accuracy_evidence": {
                    "holdout_mae_usd": float(row.MAE),
                    "holdout_rmse_usd": float(row.RMSE),
                    "holdout_medae_usd": float(row.MedAE),
                    "holdout_r2": None if pd.isna(row.R2) else float(row.R2),
                },
                "fit_diagnosis": "frozen_variant",
                "stability_evidence": "See matched outer-fold variant evidence.",
                "runtime_evidence": "See runtime_summary.csv.",
                "decision": "comparison_only_no_automatic_promotion",
                "limitation": "One-time holdout evidence cannot change the frozen family decision.",
                "next_action": "Review alongside CV, exposure and operational evidence.",
                "evidence_refs": [f"holdout_metrics.csv#model_role_id={role}"],
            }
        )
    return conclusions


def build_agent_summary(
    *,
    run_id: str,
    summary: pd.DataFrame,
    selection: dict[str, Any],
    fold_summary: pd.DataFrame,
    conclusions: list[dict[str, Any]],
) -> dict[str, Any]:
    fastest = summary.sort_values("total_cv_fit_wall_s").iloc[0]
    return {
        "schema_version": "training-validation-agent/v1",
        "run_id": run_id,
        "execution_status": "complete",
        "lowest_cv_mae_model_id": selection["lowest_mae_model"],
        "selected_family_id": selection["selected_family"],
        "fastest_fit_model_id": str(fastest.model),
        "folds": fold_summary[
            [
                "fold_id",
                "train_period_min",
                "train_period_max",
                "validation_period_min",
                "train_rows",
                "validation_rows",
                "not_used_yet_rows",
            ]
        ].to_dict("records"),
        "model_conclusions": conclusions,
        "requires_user_approval": True,
        "safe_next_action": "Review evidence; do not automatically deploy or retrain.",
    }


def _json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return frame.astype(object).where(pd.notna(frame), None).to_dict("records")


def build_ui_summary(
    *,
    run_id: str,
    manifest_sha256: str | None,
    summary: pd.DataFrame,
    selection: dict[str, Any],
    fold_summary: pd.DataFrame,
    conclusions: list[dict[str, Any]],
) -> dict[str, Any]:
    candidate_conclusions = [
        record for record in conclusions if record["model_role_id"] in CANDIDATE_ROLES
    ]
    final_conclusions = [record for record in conclusions if record["model_role_id"] in FINAL_ROLES]
    return {
        "schema_version": "training-validation-ui/v1",
        "run_id": run_id,
        "method_version": "expanding-monthly-v1",
        "source_manifest": {
            "path": "manifest.json",
            "sha256": manifest_sha256,
            "sha256_reason": (
                None if manifest_sha256 else "self_referential_manifest_hash_not_embedded"
            ),
        },
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "available_sections": ["page04.fold_method", "page04.quality", "page05.variants"],
        "page04": {
            "fold_method": {
                "status": "available",
                "rows": _json_records(fold_summary),
                "source": "fold_summary.csv",
            },
            "quality": {
                "status": "available",
                "rows": _json_records(summary),
                "source": "candidate_summary.csv",
            },
            "conclusion": {
                "lowest_cv_mae_model_id": selection["lowest_mae_model"],
                "selected_family_id": selection["selected_family"],
            },
            "model_conclusions": candidate_conclusions,
        },
        "page05": {"model_conclusions": final_conclusions},
        "warnings": ["Offline evidence only; existing UI pages are not automatically updated."],
        "downloads": [
            {"path": "fold_summary.csv", "kind": "table"},
            {"path": "model_conclusions.json", "kind": "json"},
        ],
    }


def evaluate_outcome(
    holdout_metrics: dict[str, float] | None, thresholds: dict[str, float | None]
) -> dict[str, Any]:
    if holdout_metrics is None:
        return {"status": "Blocked", "reason": "authorized holdout evidence is unavailable"}
    required = ("max_holdout_mae_usd", "max_holdout_rmse_usd")
    if any(thresholds.get(name) is None for name in required):
        return {"status": "Inconclusive", "reason": "scientific thresholds were not configured"}
    criteria = {
        "MAE": {
            "observed": float(holdout_metrics["MAE"]),
            "limit": float(thresholds["max_holdout_mae_usd"]),
        },
        "RMSE": {
            "observed": float(holdout_metrics["RMSE"]),
            "limit": float(thresholds["max_holdout_rmse_usd"]),
        },
    }
    for criterion in criteria.values():
        criterion["verdict"] = (
            "pass" if criterion["observed"] <= criterion["limit"] else "fail"
        )
    status = "Good" if all(item["verdict"] == "pass" for item in criteria.values()) else "Bad"
    return {"status": status, "criteria": criteria}


def build_training_log(
    *,
    run_context: dict[str, str],
    requested_shares: dict[str, float],
    actual_counts: dict[str, int],
    actual_shares: dict[str, float],
    fold_summary: pd.DataFrame,
    candidate_fold_metrics: pd.DataFrame,
    candidate_summary: pd.DataFrame,
    fit_diagnostic_rows: pd.DataFrame,
    ablation: pd.DataFrame,
    fold_importance: pd.DataFrame,
    importance_drift: pd.DataFrame,
    selection: dict[str, Any],
    tuning_trials: pd.DataFrame,
    tuning_summary: dict[str, Any],
    variant_metrics: pd.DataFrame,
    holdout_metrics: pd.DataFrame,
    holdout_predictions: pd.DataFrame,
    encoded_importance: pd.DataFrame,
    permutation_importance: pd.DataFrame,
    subgroups: pd.DataFrame,
    uncertainty: list[dict[str, Any]],
    outcome: dict[str, Any],
    operational: dict[str, Any],
    fit_count: int,
) -> str:
    """Render a deterministic English method log from the persisted evidence values."""

    def number(value: Any, digits: int = 3) -> str:
        if value is None or pd.isna(value):
            return "not_available"
        return f"{float(value):.{digits}f}"

    lines = [
        "# Training validation execution log",
        "Evidence-derived method log. Every number below comes from the files in this run pack.",
        "",
        "## 5W1H",
    ]
    lines.extend(f"{key.title()}: {run_context[key]}" for key in ("who", "what", "when", "where", "why", "how"))
    lines.extend(
        [
            "",
            "## Step 1 — Partition and leakage boundary",
            (
                "Requested split: TRAIN "
                f"{100 * requested_shares['train']:.2f}% | EVALUATION_HOLDOUT "
                f"{100 * requested_shares['evaluation_holdout']:.2f}% | INFERENCE_RESERVE "
                f"{100 * requested_shares['inference_reserve']:.2f}%"
            ),
        ]
    )
    for partition in ("TRAIN", "EVALUATION_HOLDOUT", "INFERENCE_RESERVE"):
        lines.append(
            f"Actual partition | name={partition} | rows={int(actual_counts[partition])} | "
            f"share={100 * float(actual_shares[partition]):.2f}%"
        )
    lines.append(
        "The split uses complete chronological months, so actual percentages may differ from the requested targets."
    )
    lines.append("The inference reserve is excluded from fitting, tuning, and evaluation.")
    lines.extend(["", "## Step 2 — Feature engineering"])
    lines.append(
        "Feature engineering uses the approved raw feature policy. Preprocessing is fitted inside each model pipeline and each temporal fold."
    )
    lines.append("Target and row identifiers are excluded from model features.")
    lines.extend(["", "## Step 3 — Expanding monthly validation folds"])
    ordered_folds = fold_summary.copy()
    ordered_folds["_scope_order"] = ordered_folds["scope"].map({"outer": 0, "inner": 1})
    for _, row in ordered_folds.sort_values(
        ["_scope_order", "fold_id", "ordinal"]
    ).iterrows():
        parent = "TRAIN" if pd.isna(row.parent_fold_id) else row.parent_fold_id
        lines.append(
            f"Fold | id={row.fold_id} | scope={row.scope} | parent={parent} | "
            f"train={row.train_period_min}..{row.train_period_max} ({int(row.train_rows)} rows) | "
            f"validate={row.validation_period_min}..{row.validation_period_max} "
            f"({int(row.validation_rows)} rows) | later_unused={int(row.not_used_yet_rows)} | "
            f"chronology={'pass' if bool(row.chronological_order_ok) else 'fail'} | "
            f"row_overlap={int(row.row_overlap_count)}"
        )
    lines.extend(["", "## Step 4 — Five-model training and evaluation"])
    lines.append("Classification metrics are not applicable because annual_salary_usd is a continuous regression target.")
    for _, row in candidate_fold_metrics.sort_values(["fold_ordinal", "model"]).iterrows():
        lines.append(
            f"Candidate fold | model={row.model} | fold={row.fold_id} | "
            f"train_MAE_usd={number(row.train_MAE)} | validation_MAE_usd={number(row.validation_MAE)} | "
            f"validation_RMSE_usd={number(row.validation_RMSE)} | "
            f"validation_MedAE_usd={number(row.validation_MedAE)} | "
            f"validation_R2={number(row.validation_R2)} | fit_wall_s={number(row.fit_wall_s, 6)}"
        )
    for _, row in candidate_summary.sort_values("cv_mae_mean_usd").iterrows():
        lines.append(
            f"Candidate summary | model={row.model} | CV_MAE_mean_usd={number(row.cv_mae_mean_usd)} | "
            f"CV_MAE_sd_usd={number(row.cv_mae_sd_usd)} | CV_RMSE_mean_usd={number(row.cv_rmse_mean_usd)} | "
            f"CV_MedAE_mean_usd={number(row.cv_medae_mean_usd)} | CV_R2_mean={number(row.cv_r2_mean)} | "
            f"total_fit_wall_s={number(row.total_cv_fit_wall_s, 6)}"
        )
    for _, row in fit_diagnostic_rows.sort_values(["model", "fold_id"]).iterrows():
        lines.append(
            f"Fit diagnosis | model={row.model} | fold={row.fold_id} | indication={row.fold_indication} | "
            f"aggregate={row.aggregate_indication} | normalized_MAE_gap={number(row.normalized_mae_gap)}"
        )
    lines.append(
        f"Family selection | lowest_CV_MAE={selection['lowest_mae_model']} | "
        f"selected={selection['selected_family']} | rule={selection['selection_method']} | "
        "runtime_used_for_selection=false"
    )
    lines.extend(["", "## Step 5 — Feature-family evidence"])
    for _, row in ablation.sort_values(["removed_family", "fold_id"]).iterrows():
        lines.append(
            f"Feature-family ablation | removed={row.removed_family} | fold={row.fold_id} | "
            f"validation_MAE_usd={number(row.validation_MAE)} | "
            f"delta_vs_full_usd={number(row.mae_delta_vs_full)}"
        )
    for _, row in fold_importance.sort_values(["fold_id", "family"]).iterrows():
        lines.append(
            f"Fold importance | fold={row.fold_id} | family={row.family} | "
            f"normalized_importance={number(row.normalized_importance, 6)}"
        )
    for _, row in importance_drift.sort_values("fold_id").iterrows():
        lines.append(
            f"Importance drift | previous={row.previous_fold_id} | fold={row.fold_id} | "
            f"half_L1={number(row.half_l1_drift, 6)}"
        )
    lines.extend(["", "## Step 6 — Conditional Random Forest GridSearchCV"])
    if tuning_trials.empty:
        for context, summary in sorted(tuning_summary.items()):
            lines.append(
                f"Grid search | context={context} | status={summary['status']} | reason={summary['reason']}"
            )
    else:
        for _, row in tuning_trials.sort_values(["search_id", "rank", "trial_ordinal"]).iterrows():
            lines.append(
                f"Grid candidate | search={row.search_id} | trial={row.trial_id} | "
                f"n_estimators={int(row.n_estimators)} | max_depth={row.max_depth} | "
                f"mean_MAE_usd={number(row.mae_mean)} | sd_MAE_usd={number(row.mae_sd)} | "
                f"mean_R2={number(row.r2_mean)} | rank={int(row['rank'])} | status={row.status}"
            )
        for context, summary in sorted(tuning_summary.items()):
            lines.append(
                f"Grid winner | context={context} | status={summary['status']} | "
                f"parameters={json.dumps(summary['winner'], sort_keys=True)} | fits={int(summary['actual_fit_count'])}"
            )
    lines.extend(["", "## Step 7 — Frozen variants and one-time holdout evaluation"])
    for _, row in variant_metrics.sort_values(["fold_id", "variant"]).iterrows():
        lines.append(
            f"Variant fold | fold={row.fold_id} | variant={row.variant} | "
            f"validation_MAE_usd={number(row.validation_MAE)} | validation_RMSE_usd={number(row.validation_RMSE)} | "
            f"validation_MedAE_usd={number(row.validation_MedAE)} | validation_R2={number(row.validation_R2)}"
        )
    for _, row in holdout_metrics.sort_values("model_role_id").iterrows():
        lines.append(
            f"Holdout metric | model={row.model_role_id} | rows={int(row.rows)} | MAE_usd={number(row.MAE)} | "
            f"RMSE_usd={number(row.RMSE)} | MedAE_usd={number(row.MedAE)} | R2={number(row.R2)}"
        )
    for model, rows in holdout_predictions.groupby("model_role_id", sort=True):
        lines.append(
            f"Residual evidence | model={model} | rows={len(rows)} | "
            f"mean_residual_usd={number(rows['residual'].mean())} | "
            f"median_absolute_error_usd={number(rows['absolute_error'].median())}"
        )
    lines.extend(["", "## Step 8 — Explainability, subgroup checks, and uncertainty"])
    encoded = encoded_importance.groupby(
        ["model_role_id", "raw_feature"], as_index=False
    ).agg(
        importance=("importance", lambda values: values.sum(min_count=1)),
        method=("method", "first"),
    )
    for _, row in encoded.sort_values(
        ["model_role_id", "importance"], ascending=[True, False], na_position="last"
    ).iterrows():
        lines.append(
            f"Encoded importance | model={row.model_role_id} | method={row.method} | "
            f"raw_feature={row.raw_feature} | importance={number(row.importance, 6)}"
        )
    permutation = permutation_importance.groupby(["kind", "name"], as_index=False)["mae_increase"].mean()
    for _, row in permutation.sort_values(["kind", "mae_increase"], ascending=[True, False]).iterrows():
        lines.append(
            f"Permutation importance | kind={row.kind} | name={row['name']} | mean_MAE_increase_usd={number(row.mae_increase)}"
        )
    for _, row in subgroups.sort_values(["model_role_id", "subgroup", "value"]).iterrows():
        lines.append(
            f"Subgroup error | model={row.model_role_id} | subgroup={row.subgroup} | value={row.value} | "
            f"support={int(row.support)} | small_sample={str(bool(row.small_sample)).lower()} | MAE_usd={number(row.MAE)}"
        )
    for item in uncertainty:
        lines.append(
            f"Empirical q90 band | model={item['model_role_id']} | q90_absolute_error_usd={number(item['q90_absolute_error_usd'])} | "
            f"same_population_coverage={number(item['same_population_coverage'], 6)} | basis={item['basis']}"
        )
    lines.extend(["", "## Step 9 — Status and conclusion"])
    lines.append(
        f"Final conclusion | scientific_status={outcome['status']} | operational_status={operational['operational_status']} | fits={fit_count}"
    )
    lines.append(
        "Good means the approved scientific thresholds passed and the model can proceed to human deployment review. Bad means return to feature engineering, training, evaluation, and bounded tuning."
    )
    lines.append("Monitoring and retraining require separately approved future data; this run does not activate deployment.")
    return "\n".join(lines) + "\n"


def render_report(
    *, run_context: dict[str, str], fold_explanation: str, model_conclusions: list[dict[str, Any]]
) -> str:
    required = ("who", "what", "when", "where", "why", "how")
    if any(not run_context.get(key) for key in required):
        raise ValueError("report context must answer 5W1H")
    lines = ["# Training Validation Report", "", "## 5W1H"]
    for key in required:
        lines.append(f"- **{key.title()}**: {run_context[key]}")
    lines.extend(
        [
            "",
            "Classification metrics: not applicable — the current target is continuous salary regression.",
            "",
            fold_explanation.rstrip(),
            "",
            "## Model conclusions",
        ]
    )
    for conclusion in model_conclusions:
        lines.extend(
            [
                "",
                f"### {conclusion['model_role_id']}",
                f"- Finding: {conclusion['finding']}",
                f"- Evidence: {', '.join(conclusion['evidence_refs'])}",
                f"- Limitation: {conclusion['limitation']}",
                f"- Next action: {conclusion['next_action']}",
            ]
        )
    return "\n".join(lines) + "\n"


def write_report_charts(
    directory: Path,
    summary: pd.DataFrame,
    *,
    holdout_predictions: pd.DataFrame,
    importance: pd.DataFrame,
) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    comparison = go.Figure(
        go.Bar(x=summary["model"], y=summary["cv_mae_mean_usd"], name="CV MAE")
    )
    comparison.update_layout(title="Five-model temporal CV MAE", yaxis_title="USD/year")
    tradeoff = go.Figure(
        go.Scatter(
            x=summary["total_cv_fit_wall_s"],
            y=summary["cv_mae_mean_usd"],
            mode="markers+text",
            text=summary["model"],
        )
    )
    tradeoff.update_layout(
        title="Accuracy and fit-cost evidence",
        xaxis_title="Total CV fit wall time (s)",
        yaxis_title="CV MAE (USD/year)",
    )
    diagnostics = go.Figure(
        go.Scatter(
            x=holdout_predictions["actual"],
            y=holdout_predictions["predicted"],
            mode="markers",
            customdata=holdout_predictions[["residual"]],
            hovertemplate="actual=%{x}<br>predicted=%{y}<br>residual=%{customdata[0]}",
        )
    )
    diagnostics.update_layout(
        title="Frozen holdout actual versus predicted",
        xaxis_title="Actual annual salary (USD/year)",
        yaxis_title="Predicted annual salary (USD/year)",
    )
    importance_chart = go.Figure(
        go.Bar(x=importance["mae_increase"], y=importance["name"], orientation="h")
    )
    importance_chart.update_layout(
        title="Holdout permutation MAE increase",
        xaxis_title="MAE increase (USD/year)",
        yaxis_title="Feature or family",
    )
    outputs = [
        (directory / "model_comparison.html", comparison),
        (directory / "accuracy_runtime_tradeoff.html", tradeoff),
        (directory / "holdout_diagnostics.html", diagnostics),
        (directory / "feature_importance.html", importance_chart),
    ]
    for path, figure in outputs:
        figure.write_html(path, include_plotlyjs=True, full_html=True)
    return [path for path, _ in outputs]
