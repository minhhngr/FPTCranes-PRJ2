"""Pure human, agent, and future-UI views over validated training evidence."""

from __future__ import annotations

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
